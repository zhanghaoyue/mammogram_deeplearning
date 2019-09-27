#!/usr/bin/python
# -*- coding: utf-8 -*-
# =============================================================================
__author__ = "Zichen Wang"
__maintainer__ = "Zichen Wang"
__credits__ = ["Zichen Wang", ]
__version__ = "1.0.0"
__email__ = "zcwang0702@ucla.edu"
__date__ = 5 / 30 / 19
__description__ = ""
# =============================================================================


if __name__ == '__main__':
    import json

    import matplotlib.pyplot as plt
    import torch
    import numpy as np
    from skimage import transform
    from PIL import Image
    import graph.model.model as module_arch
    from dataloader.base_data_loaders import basic_transform_img

    model = module_arch.AttU_Net(img_ch=1, output_ch=2, attention_map=True).cuda()
    config = json.load(open('./config_test.json'))
    resume_path = '../checkpoint/Zichen_model.pth'
    checkpoint = torch.load(resume_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    image = Image.open('./asset/0a4c9c21d3666deb33fa50d6991e9d6b.png')
    image_tensor = basic_transform_img(image).unsqueeze(0).cuda()

    prob, attention_map = model(image_tensor)

    attention_map_numpy = attention_map.data.cpu().numpy()[0]
    attention_map_reshape = transform.resize(attention_map_numpy,
                                             np.array(image).shape,
                                             mode='edge',
                                             anti_aliasing=False,
                                             anti_aliasing_sigma=None,
                                             preserve_range=True,
                                             order=0)
    print(prob)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.subplot(1, 2, 2)
    plt.imshow(attention_map_reshape, cmap='jet')
    plt.show()

    plt.figure()
    plt.imshow(image)
    plt.imshow(attention_map_reshape, cmap='jet', alpha=0.5)
    plt.show()
    import pdb
    pdb.set_trace()
