import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data_utils
from torchvision import transforms
import get_dataset

#############Set path's to images, clinical dataframe, home folder##############

mg_images = '/home/d.gordon/image_athena_no_implant/'
bc_clinical_sub = pd.read_csv('/home/d.gordon/label_athena_no_implant.csv',low_memory=False)
d_path = '/home/d.gordon/'


##############################Balance Data#####################################

bc_clinical_subset_1 = bc_clinical_sub.loc[bc_clinical_sub['cancer_label']==1]
bc_clinical_subset_0 = bc_clinical_sub.loc[bc_clinical_sub['cancer_label']==0].sample(n=len(bc_clinical_subset_1),random_state=546)
bc_clinical_subset = bc_clinical_subset_1.append(bc_clinical_subset_0)

###########################Preprocess Images###################################


## read in image file and apply image processing

for row in bc_clinical_subset.itertuples(index=False):
    try:
        img = cv2.imread(mg_images+row.filename+'.jpg')

        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        grayscale[grayscale>254] = 0

        ret, th = cv2.threshold(grayscale, 0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        bbox = cv2.boundingRect(th)

        x,y,w,h = bbox

        foreground = img[y:y+h, x:x+w]

        cv2.imwrite(d_path+'preprocessed5/'+row.filename+'.png',foreground)
    except:
        continue

#############################Extract Patches##################################

images = []
patchess = []
for filename in os.listdir(d_path+ 'preprocessed5/'):
    img = cv2.imread(d_path + 'preprocessed5/'+filename)
    patches = image.extract_patches_2d(img,max_patches=50,patch_size=(128,128))
    images.append(img)
    patchess.append(patches)

# Reset clinical dataset index before data dictionary
bc_clinical_subset.reset_index(inplace=True,drop=True)

##############################Data Dictionary##################################
data = {}

for i in range(0,len(bc_clinical_subset)):
    label = bc_clinical_subset.cancer_label[i]
    patient = bc_clinical_subset.filename[i]
    for patch in patchess:
        patch_level_of_images = patch #all images(580) with 50 patches per image
        for patches in patch_level_of_images:
            patch_level_of_patch = patches #one image(1) with 50 patches per image

            # load data
            img = patch_level_of_patch # 50 patches per image value.

            data[patient] = {
            'imgs':img,
            'label':label
                    }

###########################Data Transform######################################
transform_train = transforms.Compose([
transforms.ToPILImage(),
transforms.RandomRotation(90),
#transforms.RandomResizedCrop(128),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize(mean=[0.5,0.5,0.5],
                     std=[0.5,0.5,0.5])
    ])

transform_test = transforms.Compose([
transforms.ToPILImage(),
#transforms.Resize(128),
transforms.ToTensor(),
transforms.Normalize(mean=[0.5,0.5,0.5],
                     std=[0.5,0.5,0.5])
    ])

dataset_transform = {'train':transform_train, 'test':transform_test}
                

#############################Folds############################################
num_folds = 10
# parameters
folds = []
output = '%s/indices_%s_fold.pkl' %(d_path,num_folds)
# check if the folder exist
if not os.path.exists(output):
# create the stratified fold
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=102)
    for train_index,test_index in skf.split(bc_clinical_subset.filename,bc_clinical_subset.cancer_label):
            folds.append(bc_clinical_subset.filename[test_index][:-4].tolist())
    # save the fold
    with open(output, 'wb') as l:
        pickle.dump(folds,l)
else:
    # load
    with open(output, 'rb') as l:
        folds = pickle.load(l)

# gets datasets, labels, dataloaders
for fold_index in range(0,len(folds)):
    datasets,labels,dataloaders = get_train_test_set(folds, fold_index, data, dataset_transform)

#############################Form Bags#########################################
class MGbags(data_utils.Dataset):
    """
Pytorch dataset object that loads balanced mammogram dataset in bag form.

Implementation influenced by Ilse et al. (2018).

    """
    def __init__(self, target_number=1, number_of_patches=50, variance_of_number_patches=0, num_bag=580, seed=1, train=True):
        """

:target number: the desired bag label number. In this project, use 1, as bags have a positive label (1)
if they contain at least 1 cancerous patch or a negative label (0) if they do not contain any cancerous patches.
The aim of our model is to predict this target number (the bag label).
:number of patches: the desired number of patches.  In this project, we extracted 50 patches of size 128 x 128 from the image after it was segmented.  These number of patches (50) are contained within one bag.
The form bag function we adapted does not generally support more than 10 instances per bag; however, it will handle up to 50.
:variance of number patches: the desired variance for the number of patches.  In this project, we set to 0
because we have a fixed number of patches extracted from the image which are contained within one bag.
:number of bags: the number of bags in total model, which is the combined number of training bags and testing bags.
:seed: set random seed.
:train: train.
:num_bags_train: This contains the number of bags in the training model, which can also be interpreted as the number of images in your training set before having extracted patches.
For balanced athena dataset set to 522.
:num_bags_test: This contains the number of bags in the test model, which can also be interpreted as
the number of images in your test set before having extracted patches.  For balanced athena dataset set to 58.


Note: In future, instead of hardcoding the inputs for the above parameters, consider using something like int(input('enter desired number')),
may be more user friendly and allow for more flexibility as you work with different datasets.
        """
        
        self.target_number = target_number
        self.number_of_patches = number_of_patches # number of patches
        self.variance_of_number_patches = variance_of_number_patches #consider to set to zero because use fixed number of patches
        self.num_bag = num_bag
        self.seed = seed
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 522  # enter number of images in the training set (prior to extracting patches)
        self.num_in_test =  58 # enter number of images in the test set (prior to extracting patches)

        if self.train:
            self.train_bags_list, self.train_labels_list = self._form_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._form_bags()

    def _form_bags(self):
        if self.train:

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in dataloaders['train']:
                numbers = batch_data[1]
                labels = batch_data[2]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.number_of_patches, self.variance_of_number_patches, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_train, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        else:

            bags_list = []
            labels_list = []
            valid_bags_counter = 0
            label_of_last_bag = 0

            for batch_data in dataloaders['test']:
                numbers = batch_data[1]
                labels = batch_data[2]

            while valid_bags_counter < self.num_bag:
                bag_length = np.int(self.r.normal(self.number_of_patches, self.variance_of_number_patches, 1))
                if bag_length < 1:
                    bag_length = 1
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))
                labels_in_bag = labels[indices]

                if (self.target_number in labels_in_bag) and (label_of_last_bag == 0):
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[indices])
                    label_of_last_bag = 1
                    valid_bags_counter += 1
                elif label_of_last_bag == 1:
                    index_list = []
                    bag_length_counter = 0
                    while bag_length_counter < bag_length:
                        index = torch.LongTensor(self.r.randint(0, self.num_in_test, 1))
                        label_temp = labels[index]
                        if label_temp.numpy()[0] != self.target_number:
                            index_list.append(index)
                            bag_length_counter += 1

                    index_list = np.array(index_list)
                    labels_in_bag = labels[index_list]
                    labels_in_bag = labels_in_bag >= self.target_number
                    labels_list.append(labels_in_bag)
                    bags_list.append(numbers[index_list])
                    label_of_last_bag = 0
                    valid_bags_counter += 1
                else:
                    pass

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [max(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [max(self.test_labels_list[index]), self.test_labels_list[index]]

        return bag, label


if __name__ == "__main__":
    to_pil = transforms.Compose([transforms.ToPILImage()])

    kwargs = {}
    batch_size = 1

    train_loader = data_utils.DataLoader(MGbags(target_number= 1, number_of_patches= 50,variance_of_number_patches= 0,
                                                num_bag= 522,
                                                seed= 10,
                                                   train=True),
                                         batch_size=batch_size,
                                         shuffle=False, **kwargs)

    test_loader = data_utils.DataLoader(MGbags(target_number= 1, number_of_patches= 50,variance_of_number_patches= 0,num_bag= 58, seed= 10,train=False),batch_size=batch_size, shuffle=False, **kwargs)

    len_bag_list = []
    MG_bags_train = 0
    for batch_idx, data in enumerate(train_loader):
        plot_data = data[0].squeeze(0)
        len_bag_list.append(int(plot_data.size()[0]))
        # plot_data = data[0].squeeze(0)
        # num_instances = int(plot_data.size()[0])
        # print(data[1][0])
        # for i in range(num_instances):
        #     plt.subplot(num_instances, 1, i + 1)
        #     to_pil(plot_data[i, :, :, :]).show()
        # plt.show()
        if data[1][0][0] == 1:
            MG_bags_train += 1
    #print('number of training MG bags with 1(s): ', MG_bags_train)
    #print('total number of training MG bags:', len(train_loader))
    #print('number of patches per bag(mean,min,max):', np.mean(len_bag_list), np.min(len_bag_list), np.max(len_bag_list))

    len_bag_list = []
    MG_bags_test = 0
    for batch_idx, data in enumerate(test_loader):
        plot_data = data[0].squeeze(0)
        len_bag_list.append(int(plot_data.size()[0]))
        if data[1][0][0] == 1:
            MG_bags_test += 1
    #print('number of test MG bags with 1(s): ', MG_bags_test)
    #print('total number of test MG bags:', len(test_loader))
    #print('number of patches per bag(mean,min,max):', np.mean(len_bag_list), np.min(len_bag_list), np.max(len_bag_list))
