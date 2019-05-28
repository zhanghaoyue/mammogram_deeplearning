import schedule
from shutil import copyfile
import os.path
from flask import Flask, abort, make_response, render_template, url_for, request
from io import BytesIO
import openslide
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from optparse import OptionParser
import pymssql
import sqlalchemy
from PIL import Image
import atexit


DEEPZOOM_SLIDE = None
DEEPZOOM_MASK = None
DEEPZOOM_FORMAT = 'jpeg'
DEEPZOOM_TILE_SIZE = 256
DEEPZOOM_OVERLAP = 1
DEEPZOOM_LIMIT_BOUNDS = True
DEEPZOOM_TILE_QUALITY = 40
SLIDE_NAME = 'slide'
Image.MAX_IMAGE_PIXELS = None


app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_envvar('DEEPZOOM_TILER_SETTINGS', silent=True)

# SQL configurations string
engine = sqlalchemy.create_engine('mssql+pymssql://IDRReadOnly:yCy*X&c5LbOal5Y7G83$@10.32.159.137:1433/Aperio')


class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')


@app.before_first_request
def preload_slide():
    app.config['DEEPZOOM_SLIDE'] = '/home/harry/1302.svs'
    slidefile = app.config['DEEPZOOM_SLIDE']
    #maskfile = app.config['DEEPZOOM_MASK']
    if slidefile is None:
        raise ValueError('No slide file specified')
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    osr = OpenSlide(slidefile)
    #mask = OpenSlide(maskfile)
    app.slide = DeepZoomGenerator(osr, **opts)
    #app.mask = DeepZoomGenerator(mask, **opts)
    app.slide.name = app.config['DEEPZOOM_SLIDE'][-8:-4]
    app.slide.properties = osr.properties
    app.slide_path = slidefile
    try:
        mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
        app.slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
    except (KeyError, ValueError):
        app.slide.mpp = 0
    return app.slide


@app.route('/image_metadata', methods=['POST'])
def get_metadata():

    if not request.json:
        abort(400)
    accession_number = request.json['accession_number']
    if accession_number:
        query = "select ROW_NUMBER() OVER(ORDER BY (SELECT 100)) as row, " \
                "substring(CompressedFileLocation,28, len(CompressedFileLocation)) as path " \
                "from dbo.Image where ParentId IN " \
                "(select Id from dbo.Slide WHERE dbo.Slide.ParentId IN " \
                "(select Id from dbo.Specimen WHERE ColumnUCLASpecimen ='" + str(accession_number) + "'))"

        with engine.connect() as con:
            query_result = con.execute(query)
            # dict to store all paths for each accession number
            resultset = [dict(row) for row in query_result]
            for i, string in enumerate(resultset):
                resultset[i]['path'] = resultset[i]['path'].replace("\\", "/")

            if resultset is None:
                abort(404)
            else:
                app.slide_path = resultset
            src = '/media/' + resultset[0]['path']
            dst = '/home/temp/' + resultset[0]['path'].split('/')[-1]
            copyfile(src, dst)
            app.config['DEEPZOOM_SLIDE'] = dst
    try:
        load_slide()
        return 'OK'
    except(KeyError, ValueError):
        abort(404)


@app.route('/load_slide')
def load_slide():
    slidefile = app.config['DEEPZOOM_SLIDE']
    # maskfile = app.config['DEEPZOOM_MASK']
    if slidefile is None:
        raise ValueError('No slide file specified')
    config_map = {
        'DEEPZOOM_TILE_SIZE': 'tile_size',
        'DEEPZOOM_OVERLAP': 'overlap',
        'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
    }
    opts = dict((v, app.config[k]) for k, v in config_map.items())
    osr = OpenSlide(slidefile)
    # mask = OpenSlide(maskfile)
    app.slide = DeepZoomGenerator(osr, **opts)
    # app.mask = DeepZoomGenerator(mask, **opts)
    app.slide.name = app.config['DEEPZOOM_SLIDE'].split('/')[-1]
    app.slide.properties = osr.properties
    try:
        mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
        mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
        app.slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
    except (KeyError, ValueError):
        app.slide.mpp = 0
    return app.slide


@app.route('/update_path', methods=['POST'])
def update_path():
    selected_path = request.form['selected_path']
    dst = '/home/temp' + selected_path.split('/')[-1]
    if not os.path.isfile(dst):
        src = '/media/' + selected_path
        copyfile(src, dst)
    app.config['DEEPZOOM_SLIDE'] = dst
    print(app.config['DEEPZOOM_SLIDE'])
    #pdb.set_trace()
    try:
        app.slide = load_slide()
        slide_url = url_for('dzi', path=app.slide.name)
        print(slide_url)
        return render_template('demo.html', slide_url=slide_url, properties=app.slide.properties,
                               slide_path=app.slide_path, slide_mpp=app.slide.mpp)
    except(KeyError, ValueError):
        abort(404)


@app.route('/')
def index():
    slide_url = url_for('dzi', path=app.slide.name)
    # mask_url = url_for('mask', mask_path=app.slide.name+"pred")
    return render_template('demo.html', slide_url=slide_url, properties=app.slide.properties,
                           slide_path=app.slide_path, slide_mpp=app.slide.mpp)


@app.route('/<path>.dzi')
def dzi(path):
    format = app.config['DEEPZOOM_FORMAT']
    try:
        resp = make_response(app.slide.get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp
    except KeyError:
        abort(404)


@app.route('/<mask_path>.dzi')
def mask(mask_path):
    format = app.config['DEEPZOOM_FORMAT']
    try:
        resp = make_response(app.mask.get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp
    except KeyError:
        abort(404)


@app.route('/<path>_files/<int:level>/<int:col>_<int:row>.<format>')
def tile(path,level, col, row, format):
    format = format.lower()
    if format != 'jpeg':
        # Not supported by Deep Zoom
        abort(404)
    try:
        tile = app.slide.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


'''
@app.route('/<mask_path>_files/<int:level>/<int:col>_<int:row>.<format>')
def mask_tile(mask_path,level, col, row, format):
    format = format.lower()
    if format != 'jpeg':
        # Not supported by Deep Zoom
        abort(404)
    try:
        mask_tile = app.mask.get_tile(level, (col, row))
    except ValueError:
        # Invalid level or coordinates
        abort(404)
    buf = PILBytesIO()
    mask_tile.save(buf, format, quality=app.config['DEEPZOOM_TILE_QUALITY'])
    resp = make_response(buf.getvalue())
    resp.mimetype = 'image/%s' % format
    return resp


def slugify(text):
    text = normalize('NFKD', text.lower()).encode('ascii', 'ignore').decode()
    return re.sub('[^a-z0-9]+', '-', text)
'''


def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()


@app.route('/shutdown', methods=['POST'])
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


def clean_tempfolder():
    folder = '/home/temp'
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            # elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


schedule.every().hour.do(clean_tempfolder)


if __name__ == '__main__':
    parser = OptionParser(usage='Usage: %prog [options] [slide]')
    parser.add_option('-B', '--ignore-bounds', dest='DEEPZOOM_LIMIT_BOUNDS',
                default=True, action='store_false',
                help='display entire scan area')
    parser.add_option('-c', '--config', metavar='FILE', dest='config',
                help='config file')
    parser.add_option('-d', '--debug', dest='DEBUG', action='store_true',
                help='run in debugging mode (insecure)')
    parser.add_option('-e', '--overlap', metavar='PIXELS',
                dest='DEEPZOOM_OVERLAP', type='int',
                help='overlap of adjacent tiles [1]')
    parser.add_option('-f', '--format', metavar='{svs}',
                dest='DEEPZOOM_FORMAT',
                help='image format for tiles [svs]')
    parser.add_option('-l', '--listen', metavar='ADDRESS', dest='host',
                default='10.1.122.61',
                help='address to listen on [10.1.122.61]')
    parser.add_option('-p', '--port', metavar='PORT', dest='port',
                type='int', default=80,
                help='port to listen on [80]')
    parser.add_option('-Q', '--quality', metavar='QUALITY',
                dest='DEEPZOOM_TILE_QUALITY', type='int',
                help='SVS compression quality [75]')
    parser.add_option('-s', '--size', metavar='PIXELS',
                dest='DEEPZOOM_TILE_SIZE', type='int',
                help='tile size [256]')

    (opts, args) = parser.parse_args()
    # Load config file if specified
    if opts.config is not None:
        app.config.from_pyfile(opts.config)
    # Overwrite only those settings specified on the command line
    for k in dir(opts):
        if not k.startswith('_') and getattr(opts, k) is None:
            delattr(opts, k)
    app.config.from_object(opts)
    # Set slide file

    try:
        app.config['DEEPZOOM_SLIDE'] = '/home/harry/1302.svs'
        # app.config['DEEPZOOM_MASK'] = '1302pred.tif'
    except IndexError:
        if app.config['DEEPZOOM_SLIDE'] is None:
            parser.error('No slide file specified')

    app.run(host=opts.host, port=opts.port)

