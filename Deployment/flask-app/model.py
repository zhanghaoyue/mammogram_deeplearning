import sys
import torch
from torchvision import transforms
import cv2
import code_zichen.graph.model.model as module_arch
from code_zichen.utils import preprocess
import numpy as np
from skimage import transform, io
from PIL import Image
import matplotlib.pyplot as plt


class Pytorchmodel:
    def __init__(self, segmentation_model_path, classification_model_path, classes_txt=None):

        self.segmentation_model = module_arch.AttU_Net(img_ch=1, output_ch=2).cuda()
        self.classification_model = module_arch.AttU_Net_Classification().cuda()
        
        self.segmentation_model.load_state_dict(torch.load(segmentation_model_path)['model_state_dict'])
        self.classification_model.load_state_dict(torch.load(classification_model_path)['model_state_dict'])

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

        img_tensor = basic_transform_img(img_png).unsqueeze(0).cuda()
        
        
        with torch.no_grad():
            self.segmentation_model.eval()    
            attention_map = self.segmentation_model(img_tensor)
            attention_map_numpy = attention_map.data.cpu().numpy()[0]
            attention_map_reshape = transform.resize(attention_map_numpy,
                                                     np.array(img_png).shape,
                                                     mode='edge',
                                                     anti_aliasing=False,
                                                     anti_aliasing_sigma=None,
                                                     preserve_range=True,
                                                     order=0)

            attention_map_png = Image.fromarray((attention_map_reshape*255).astype('uint8'))
            attetention_tensor = basic_transform_img(attention_map_png).unsqueeze(0).cuda()
            self.classification_model.eval()  
            prob = self.classification_model(img_tensor, attetention_tensor)

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
    segmentation_model_path = './code_zichen/checkpoint/segmentation_model.pth'
    classification_model_path = './code_zichen/checkpoint/classification_model.pth'
    model = Pytorchmodel(segmentation_model_path, classification_model_path)
    img_path = r'/home/harryzhang/Documents/athena_screen/images/fe22c324c0f00813c9ff635be1a62ed6.png'
    result = model.predict(img_path)

