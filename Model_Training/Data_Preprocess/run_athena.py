# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 12:58:43 2019

@author: YannanLin
"""

import utils_athena
import glob


################################ set parameters ################################

# set paths
path = "/data/athena_screen/images/"
image_path_list = glob.glob("%s/*.png" % path)
file_path = "/data/athena_screen/be223c_clinical.csv"
implant_list_path = "/home/yannan_lin/implant_list.csv"

# save processed images
image_athena_no_implant_path = "/home/yannan_lin/image_athena_no_implant/" #save images without implants
image_implant_raw_path = "/home/yannan_lin/image_implant_raw/" #save images with implants

# save names of processed images without implants
image_name_no_implant_path = "/home/yannan_lin/athena_no_implant_name_list/names.csv"
image_name_no_implant_path = "/home/yannan_lin/athena_no_implant_name_list/image_names_no_implant.csv"
image_name_no_implant_path_final = "/home/yannan_lin/athena_no_implant_name_list/athena_image_names_no_implant.csv"

# save only MLO images
MLO_path = '/home/yannan_lin/MLO/'

############################### load data #####################################
file_df, new_df, filename,cancer, name_cancer_dic, name_view_dic= load_data(file_path)

########################### load implant list #################################
implant_list = load_implant_list(implant_list_path)

######################### create train data ################################### 
image_list, label_list, name_list, label_list_implant, image_list_implant = create_train_data(
                                            image_path_list, implant_list, filename, name_cancer_dic,
                                            image_implant_raw_path,image_athena_no_implant_path,
                                            name_view_dic, MLO_path)

# also remove the following low quality images from name_list before training
low_quality_list = ['0d7e10e0a234abf04b2836928f71effe','0d9cf1cb4aef5882c0752a74cba059e9',
    '0fa390545014d81ce6c8ef5cb821a38e','1c3254c83c90ef8aff265c3d833e5b1a',
    '1fac5b1b81914b0dff6a2ffbea98280a','3bdbbcad0f3b93bf8530b1d5521807bc',
    '3c4aaa638e5a9064cae306b87bcbc62f','04d906c225003cca0f3ea65785ec4b73',
    '4d288d72998565fb39c2efc503f46c6e','05bfcfa7e4ae1314b5bb792c34c8e665',
    '5bec60f0c9b397e46f92756883e078d7','5dc0917ca02232d04e44c12a8c7e2683',
    '5e4d282dcbb938299f141620d4926e96','6c780254e28267a609f2b9f2193bef15',
    '8a56c316176ac5b04007eaacaffac44a','9d9aacc67badd2a6eb5509af40fc6e44',
    '17f50cf2298de594ac1a6807a0b235af','19ee26068dcef042d64dd11f3705dd40',
    '25a40de2891821d40f98594405d0d25c','28d69c1c6caabb08a39d11d61408f692',
    '42af982a4164f62008a6109de1681130','51c2d864709eaf46cbc045fa7a07274d',
    '53eeda2f63cde20688b78eb398277352','58a404bf6e638f2cd07dca5ce60aa438',
    '60a2e17d08211aed897c2c8739658dc2','73a30790ca360d1790876302d88a8487',
    '096f18780784e026becc057fafda51ae','297aefbb92a65c629b7f93225b276686',
    '00342cab767339aa14724fd12ff8a83d','499b19fb1e6640207f2f15beb95e2aa4',
    '645a8ee7fd0388d42825599de38a7d63','718ad0a36d28b70da98f6303982214c6',
    '846f848c82a80779505af23b6682fdeb','1819bdaf36e316c21aa6a489bdeb6b3c',
    '02331fb4003c292c65353d4a49ffc761','7019b622862fc13c77067f35354db1aa',
    '07458a5a98578b3c89687ae2da52811f','9026ef4c1becf7f99a22ffcc9e1db0c8',
    '12755f2cae02b9148dae46cfb15993c8','55184fdf061e86b6cdfc6f3711dd76a4',
    '98222a6fe360794eb55c918825bc192c','411294d14a0d295955377205739208cb',
    'a92901a698295b560c6ee9502a07af5f','b32e23ebf804bd497c0638a8eecfc21f',
    'c26ac856cb7e82a32ae987034ba40436','cfa74f395fbc41d0cd5b08049c1c8890',
    'd5a60d1e9524df29c623e17c5e44721e','d72eb690b16fba486b7960b347f28d37',
    'd89e329974aca9d56cfafdff5db5a62b','de2478e21395f582e37cc80bff542a93',
    'e2c47c1462c222cae6a8f04515e31409','e08bcc2d0bd239fe258128ba44be5da9',
    'e8bd20785e701b64890a704e4512807e','e09a8ac3e831775f4e3a7e680d18f4c9',
    'f0a709d07525f3f7ccdf30fc9a7e2618','f1b8d45f502481af236104f88542909e',
    'f32bcd610b06af5456b651760fa5909c','f229ff92e0c826f7cfd8e1ba983bf1a4',
    '4c497aac97187b301b54150a367ec706','4d288d72998565fb39c2efc503f46c6e',
    '05bfcfa7e4ae1314b5bb792c34c8e665','6aba2c5761568c774e2d8ba16841c8be',
    '6e11c84e16f856f9ed59247f105475a8','9ca8f5ff4053fca8f86097d3cfc9344e',
    '38aaa1bd6b55bfe7426081d1e1c7ffd4','215b342cf83d2eb645869646774d0531',
    '388f2be34f56ff1d57a114ec8806310f','657f30a468ab6bc7e8db42d47bb70d14',
    '846f139b63da659f06bb075cbe566ef8','1639a9088d05894d6f325709740f1567',
    '7019b622862fc13c77067f35354db1aa','7995ec5aacfdf38ba4b90f1187ad60b0',
    '40117ee3d5464fdeb7023e3e077940a1','76248b7178d0752fae49cb1dad2fce22',
    '7919555b5f506b5f03a823a3b082eca9','8355586ac07ac9ad5e1a35dd447d1c4a',
    'a07f367c3ee0fb15aab25836db9cb12a','a573e835ba718b3d5640a00dfbc9289c',
    'ad413c440d017804d271d81ef340207c','b02bd3925c3e450771f1ffb3b60bbfc9',
    'b65fff8770eec2bfc74a8faab1c8104e','b82097a7d060851cfc30748b823571c5',
    'ba331098c4c131268d4252a624a23bc6','bc25fdda6e1e7f7679c346781b9e4024',
    'c2afce8bdf17c1670e563ab7c00f2a1f','c2bbe870a6f58f80425293fe5e96ae9e',
    'c4c769bea55eb9039b67892f0eba5868','cbd0f3b954fb5e65f63ff24f609b1e20',
    'cca8ca1e8e1e92ec81920e7d83b7f02f','cdcc675f6abdc8439b1d017cca0fc143',
    'cff270d59dc56cc7a56d9a8cfd9e29da','d05c19683dd7bcf398e52a55bf4fa59d',
    'd972062cacdf3a68161f735ff3522032','d913483823aa8f6755a2e634e3f4465d',
    'da6a877ffda301ed62a8dd60eb177f61','dc2e0d3d908b9402293b53d2cc4d2015',
    'dfffef086744bdefd112e4288da70544','ea68f2ab89249d8fb1efe089fc9c1b24',
    'ed2ae45d8c0426384c1a8b64bd9c4f13','f469ef06c7a3454361275697e67c1e36',
    'f41a353f953b2737e7d7ecdae7a92292','f534943c0cd47b878626d02ab6650aed',
    'fe87d80d9f24e64c100ca84c5de27ec0']