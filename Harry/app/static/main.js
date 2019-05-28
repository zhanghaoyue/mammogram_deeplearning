App = {
	init:function(){

		var dzi_data = {{ dzi_data|default('{}')|safe }};
		var slide_path = {{slide_path|tojson}};

		// openSeadragon object;
		var viewer = new OpenSeadragon({
			id:  "view",
			tileSources: "{{ slide_url }}",
			prefixUrl: "static/images/",
			showNavigator: true,
			showRotationControl: true,
			animationTime: 0.5,
			blendTime: 0.1,
			constrainDuringPan: true,
			maxZoomPixelRatio: 2,
			minZoomLevel: 1,
			visibilityRatio: 1,
			zoomPerScroll: 2,
			//debugMode: true,
			timeout: 120000,
		});


		// To improve load times, ignore the lowest-resolution Deep Zoom
		// levels.  This is a hack: we can't configure the minLevel via
		// OpenSeadragon configuration options when the viewer is created
		// from DZI XML.
		viewer.addHandler("open", function() {
			viewer.source.minLevel = 8;
		});


		viewer.scalebar({
			xOffset: 10,
			yOffset: 10,
			barThickness: 3,
			color: '#555555',
			fontColor: '#333333',
			backgroundColor: 'rgba(255, 255, 255, 0.5)',
		});


		function open_slide(url, mpp) {
			var tile_source;
			if (dzi_data[url]) {
				// DZI XML provided as template argument (svg_to_dzi.py)
				tile_source = new OpenSeadragon.DziTileSource(
					OpenSeadragon.DziTileSource.prototype.configure(
						OpenSeadragon.parseXml(dzi_data[url]), url));
			} else {
				// DZI XML fetched from server (dzi_server.py)
				tile_source = url;
			}
			viewer.open(tile_source);
			viewer.scalebar({
				pixelsPerMeter: mpp ? (1e6 / mpp) : 0,
			});
		}


		open_slide("{{ slide_url }}", parseFloat('{{ slide_mpp }}'));


		$(".load-slide").click(function(ev) {
			$(".current-slide").removeClass("current-slide");
			$(this).parent().addClass("current-slide");
			open_slide($(this).attr('data-url'),
				parseFloat($(this).attr('data-mpp')));
			ev.preventDefault();
			window.location.reload();
		});


		// screenshot function;
		// viewer will be your OpenSeaDragon viewer object;
		viewer.screenshot({
			showOptions: true, // Default is false
			keyboardShortcut: 'p', // Default is null
			showScreenshotControl: true // Default is true
		});


		var options = {
			scale: 1000
		}


		//initialize selection
		var selection = viewer.selection(options);

		// initialize drawing overlay
		var drawing_overlay = viewer.fabricjsOverlay(options);

		var canvas = drawing_overlay.fabricCanvas('c',{
			isDrawingMode:true
		});

		fabric.Object.prototype.transparentCorners = false;

		var drawingModeEl = $('#drawing-mode'),
			drawingOptionsEl = $('#drawing-mode-options'),
			drawingColorEl = $('#drawing-color'),
			drawingShadowColorEl = $('#drawing-shadow-color'),
			drawingLineWidthEl = $('#drawing-line-width'),
			drawingShadowWidth = $('#drawing-shadow-width'),
			drawingShadowOffset = $('#drawing-shadow-offset'),
			clearEl = $('#clear-canvas');


		clearEl.click(function() { canvas.clear() });


		drawingModeEl.click(function() {
			canvas.isDrawingMode = !canvas.isDrawingMode;
			if (canvas.isDrawingMode) {
				canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
				viewer.setMouseNavEnabled(false);
				viewer.outerTracker.setTracking(false);
				drawingModeEl.text('Cancel drawing mode');
				drawingOptionsEl.show();
			}
			else {
				viewer.setMouseNavEnabled(true);
				viewer.outerTracker.setTracking(true);
				drawingModeEl.text('Enter drawing mode');
				drawingOptionsEl.hide();
			}
		});


		if (fabric.PatternBrush) {
			var vLinePatternBrush = new fabric.PatternBrush(canvas);
			vLinePatternBrush.getPatternSrc = function() {
				var patternCanvas = fabric.document.createElement('canvas');
				patternCanvas.width = patternCanvas.height = 10;
				var ctx = patternCanvas.getContext('2d');

				ctx.strokeStyle = this.color;
				ctx.lineWidth = 5;
				ctx.beginPath();
				ctx.moveTo(0, 5);
				ctx.lineTo(10, 5);
				ctx.closePath();
				ctx.stroke();

				return patternCanvas;
			};

			var hLinePatternBrush = new fabric.PatternBrush(canvas);
			hLinePatternBrush.getPatternSrc = function() {

				var patternCanvas = fabric.document.createElement('canvas');
				patternCanvas.width = patternCanvas.height = 10;
				var ctx = patternCanvas.getContext('2d');

				ctx.strokeStyle = this.color;
				ctx.lineWidth = 5;
				ctx.beginPath();
				ctx.moveTo(5, 0);
				ctx.lineTo(5, 10);
				ctx.closePath();
				ctx.stroke();

				return patternCanvas;
			};

			var squarePatternBrush = new fabric.PatternBrush(canvas);
			squarePatternBrush.getPatternSrc = function() {

				var squareWidth = 10, squareDistance = 2;

				var patternCanvas = fabric.document.createElement('canvas');
				patternCanvas.width = patternCanvas.height = squareWidth + squareDistance;
				var ctx = patternCanvas.getContext('2d');

				ctx.fillStyle = this.color;
				ctx.fillRect(0, 0, squareWidth, squareWidth);

				return patternCanvas;
			};


			var diamondPatternBrush = new fabric.PatternBrush(canvas);
			diamondPatternBrush.getPatternSrc = function() {

				var squareWidth = 10, squareDistance = 5;
				var patternCanvas = fabric.document.createElement('canvas');
				var rect = new fabric.Rect({
					width: squareWidth,
					height: squareWidth,
					angle: 45,
					fill: this.color
				});

				var canvasWidth = rect.getBoundingRect().width;

				patternCanvas.width = patternCanvas.height = canvasWidth + squareDistance;
				rect.set({ left: canvasWidth / 2, top: canvasWidth / 2 });

				var ctx = patternCanvas.getContext('2d');
				rect.render(ctx);

				return patternCanvas;
			};

			var img = new Image();
			img.src = 'static/asset/honey_im_subtle.png';

			var texturePatternBrush = new fabric.PatternBrush(canvas);
			texturePatternBrush.source = img;
		}


		$('#drawing-mode-selector').change(function() {

			if (this.value === 'hline') {
				canvas.freeDrawingBrush = vLinePatternBrush;
			}
			else if (this.value === 'vline') {
				canvas.freeDrawingBrush = hLinePatternBrush;
			}
			else if (this.value === 'square') {
				canvas.freeDrawingBrush = squarePatternBrush;
			}
			else if (this.value === 'diamond') {
				canvas.freeDrawingBrush = diamondPatternBrush;
			}
			else if (this.value === 'texture') {
				canvas.freeDrawingBrush = texturePatternBrush;
			}
			else if (this.value === 'Pencil'){
				canvas.freeDrawingBrush = new fabric.PencilBrush(canvas);
			}

			if (canvas.freeDrawingBrush) {
				canvas.freeDrawingBrush.color = drawingColorEl.value;
				canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;
				canvas.freeDrawingBrush.shadow = new fabric.Shadow({
					blur: parseInt(drawingShadowWidth.value, 10) || 0,
					offsetX: 0,
					offsetY: 0,
					affectStroke: true,
					color: drawingShadowColorEl.value,
				});
			}
		});

		drawingColorEl.change(function() {
			canvas.freeDrawingBrush.color = this.value;
		});

		drawingShadowColorEl.change(function() {
			canvas.freeDrawingBrush.shadow.color = this.value;
		});

		drawingLineWidthEl.change(function() {
			canvas.freeDrawingBrush.width = parseInt(this.value, 10) || 1;
			this.previousSibling.innerHTML = this.value;
		});

		drawingShadowWidth.change(function() {
			canvas.freeDrawingBrush.shadow.blur = parseInt(this.value, 10) || 0;
			this.previousSibling.innerHTML = this.value;
		});

		drawingShadowOffset.change(function() {
			canvas.freeDrawingBrush.shadow.offsetX = parseInt(this.value, 10) || 0;
			canvas.freeDrawingBrush.shadow.offsetY = parseInt(this.value, 10) || 0;
			this.previousSibling.innerHTML = this.value;
		});

		if (canvas.freeDrawingBrush) {
			canvas.freeDrawingBrush.color = drawingColorEl.value;
			canvas.freeDrawingBrush.width = parseInt(drawingLineWidthEl.value, 10) || 1;
			canvas.freeDrawingBrush.shadow = new fabric.Shadow({
				blur: parseInt(drawingShadowWidth.value, 10) || 0,
				offsetX: 0,
				offsetY: 0,
				affectStroke: true,
				color: drawingShadowColorEl.value,
			});
		}

		$(window).resize(function(){
			drawing_overlay.resize();
			drawing_overlay.resizecanvas();
		})


		$('#toggle-overlay').click(function(){
			viewer.addTiledImage({
				tileSource: '{{ mask_url }}',
				x: 0,
				y: 0,
				opacity: 1
			});
		})

		$('#select-slide').click(function(){
			$('#select-slide').toggleClass('selected');
			if ($('#select-slide').hasClass('selected')){
				$('#slide-list').show();
				for (var i=0;i<slide_path.length;i++) {
					$('#slide-list ul').append(
						$('<li>').append(
							$('<a>').attr('id', i).append(slide_path[i]['path'].slice(-8))
						)
					)
				}
			}
			else{
				$('#slide-list').hide();
				$('#slide-list ul li').remove();
			}
		});

		$('#slide-list ul').on('click', 'li', function(){
			var selected_string = $(this).text();
			var selected_path = "";
			for (var i=0;i<slide_path.length;i++) {
				if (slide_path[i]['path'].includes(selected_string)==true){
					selected_path = slide_path[i]['path']
				}
			}
			$.post('/update_path', {'selected_path': selected_path});
		})
	}
}

$(document).ready(function() {
	App.init();
});


