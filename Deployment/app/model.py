import sys
import torch
from torchvision import transforms
import cv2
from code_zichen.graph import model
from code_zichen.utils import preprocess
import numpy as np
from skimage import transform, io
from PIL import Image
import matplotlib.pyplot as plt

class Pytorchmodel:
    def __init__(self, model_path, img_shape, img_channel=3, classes_txt=None):
        self.img_shape = img_shape
        self.img_channel = img_channel

        self.net = model.AttU_Net(img_ch=1, output_ch=2, attention_map=True).cuda()
        checkpoint = torch.load(model_path)
        self.net.load_state_dict(checkpoint['model_state_dict'])

        if classes_txt is not None:
            with open(classes_txt, 'r') as f:
                self.idx2label = dict(line.strip().split(' ') for line in f if line)
        else:
            self.idx2label = None

    def predict(self, img_path):

        basic_transform_img = transforms.Compose([transforms.Resize((512, 512)),
                                                  transforms.ToTensor()])

        img_png = Image.open(img_path)

        # img_numpy = io.imread(img)
        #
        # img = preprocess.apply_preprocess(img_numpy)
        #
        # img_png = Image.fromarray(img)

        tensor = basic_transform_img(img_png).unsqueeze(0).cuda()

        prob, attention_map = self.net(tensor)
        attention_map_numpy = attention_map.data.cpu().numpy()[0]
        attention_map_reshape = transform.resize(attention_map_numpy,
                                                 np.array(img_png).shape,
                                                 mode='edge',
                                                 anti_aliasing=False,
                                                 anti_aliasing_sigma=None,
                                                 preserve_range=True,
                                                 order=0)

        attention_map_png = Image.fromarray((attention_map_reshape*255).astype('uint8'))
        if prob > 0.5:
            label = "Yes"
        else:
            label = "No"
            prob = 1 - prob

        sizes = np.shape(img_png)
        height = float(sizes[0])
        width = float(sizes[1])

        fig = plt.figure()
        fig.set_size_inches(width / height, 1, forward=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)

        ax.imshow(img_png, cmap='gray')
        ax.imshow(attention_map_png, cmap='jet', alpha=0.5)
        plt.savefig("overlay_png.png", dpi=height)
        plt.close()

        overlay_png = Image.open("overlay_png.png")

        result = [label, prob, img_png, attention_map_png, overlay_png]
        return result


if __name__ == '__main__':
    model_path = './code_zichen/checkpoint/Zichen_model.pth'
    model = Pytorchmodel(model_path=model_path, img_shape=[512, 512], img_channel=1)
    img_path = r'/home/harryzhang/Documents/athena_screen/images/fe22c324c0f00813c9ff635be1a62ed6.png'
    result = model.predict(img_path)

