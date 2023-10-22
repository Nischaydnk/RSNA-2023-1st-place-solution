# RSNA-2023-1st-place-solution
Code and Documentation for the first place solution in 2023 Abdominal Trauma Detection Competition hosted by RSNA on Kaggle.





# Complete Writeup
 
Here is the inference code you may refer: [link](https://www.kaggle.com/nischaydnk/rsna-super-mega-lb-ensemble) 
Our GitHub repo w/ all preprocessing + training code: [Almost finished](https://github.com/Nischaydnk/RSNA-2023-1st-place-solution)

#### **Split used:** 4 Fold GroupKFold ( Patient Level)

## **Our solution is divided into three parts:**
**Part 1:** 3D segmentation for generating masks / crops [Stage 1]
**Part 2:** 2D CNN + RNN based approach for Kidney, Liver, Spleen & Bowel [Stage 2]
**Part 3:** 2D CNN + RNN based approach for Bowel + Extravasation [Stage 2]

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2F81382ef287a512fd45d94e827c49e562%2FScreenshot%202023-10-22%20at%205.29.28%20AM.png?generation=1697935324304083&alt=media)

## **Data Preprocessing:**

Here comes the key part of our solution, we will describe it later in more depth. **Note:** *All models were trained on image size 384 x 384. We use datasets preprocessing from @TheoVeol and our data which we made rescale dicoms and applying soft-tissue windowing.*

We take a patient/study, we run a 3d segmentation model on it, it outputs masks for each slice, we make a study-level crop here based on boundaries of organs - liver, spleen, kidney & liver. 

Next, we make volumes from the patient, each volume extracted with equi-distant 96 slices for a study which is then reshaped to (32, 3, image_size, image_size) in a 2.5D manner for training CNN based models.

3 channels are formed by using the adjacent slices.

All our model takes in input in shape (2, 32, 3, height, width) and outputs it as (2, 32, n_classes) as the targets are also kept in shape (2, 32, n_classes).

To make the targets, we need 2 things, patient-level target of each organ and how much the organ is visible compared to its maximum visibility, this data is available after normalizing segmentation model masks in 0-1 based on number of positive pixels

Then we multiply targets * patient-level target for each middle slice of the sequence and that is our label

For example if a patient has label 0 for liver-injury and the liver visibility is as follows in the slice sequence

[0., 0., 0., 0.01, 0.05, 0.1, 0.23, 0.5, 0.7, 0.95, 0.99, 1., 0.95, 0.8, 0.4 …. 0. ,0., 0.]

We multiply it with label which is currently 0 results in an all zeros list as output, but if target label for liver-injury was 1, then we use the list mentioned above as our soft labels.


## **Stage2: 2.5D Approach ( 2D CNN + RNN):**

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2Fe8df4581839b1fa7dcadf68fe2a715a1%2FScreenshot%202023-10-22%20at%205.31.23%20AM.png?generation=1697935695484067&alt=media)

In stage 2, we trained our models using the volumes either based on our windowing or theo's preprocessing approach and the masks/crops generated from 3D segmentation approach. Each model is trained for multiple tasks (segmentation + classification). For all 32 sequences, we predicted slice level masks and sigmoid predictions. Further, simple maximum aggregation is applied on sigmoid predictions to fetch study level prediction used in submissions. 

For training our models, some common settings were:
- **Learning rate:** (1e-4 to 4e-4) range
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing w/ Warmup 
- **Loss:** BCE Loss for Classification, Dice Loss for segmentation



###**Auxiliary Segmentation Loss:** 
One of the key things which made our training much more stable and helped in improving scores was using auxiliary losses based on segmentation. 

Encoder was kept same for both classification & segmentation decoders,  we used two types of segmentation head:
- ***Unet based decoder*** for generating masks
- ***2D-CNN*** based head 
```
nn.Sequential(
            nn.Conv2d(nb_ft, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 4, kernel_size=1, padding=0),
        )
```
```
        self.mask_head_3 = self.get_mask_head(true_encoder.feature_info[-2]['num_chs'])
        self.mask_head_4 = self.get_mask_head(true_encoder.feature_info[-1]['num_chs'])
```
We used the feature maps generated mainly from last and 2nd last blocks of the backbones & apply dice loss on the predicted masks & true masks. This trick gave us around +0.01 to +0.03 boost in our models. We used similar technique in Covid 19 detection competition held few years back, you can also refer my solution for more detailed use of auxiliary loss & code snippets. 
[link of discussion](https://www.kaggle.com/c/siim-covid19-detection/discussion/266571)

Here is an example code for applying aux loss:
```
class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        #self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.as_tensor([2.318]).cuda()).cuda()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = smp.losses.DiceLoss(smp.losses.MULTILABEL_MODE, from_logits=True)

    def forward(self, outputs, targets, masks_outputs, masks_outputs2, masks_targets):
        loss1 = self.bce(outputs, targets.float())

        masks_outputs = masks_outputs.float()
        masks_outputs2 = masks_outputs2.float()

        masks_targets = masks_targets.float().flatten(0, 1)

        loss2 = self.dice(masks_outputs, masks_targets) + self.dice(masks_outputs2, masks_targets)


        loss = loss1 + (loss2 * 0.125) 

        return loss
```


### **Architectures used in Final ensemble:**
- Coat Lite Medium w/ GRU - [original source code](https://github.com/mlpc-ucsd/CoaT)
- Coat Lite Small w/ GRU!
- Efficientnet v2s w/ GRU [Timm]


### **Augmentations:**

We couldn't come up with several augmentations to use, but these were the ones which we used in our training.

```
        A.Perspective(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(p=0.5, limit=(-25, 25)),
```

## **Post Processing / Ensemble:**

Final ensemble for all organs model includes **multiple Coat medium and V2s based models** trained on either 4 Folds or Full data. 

For extravasation, We mainly used Coat Small and v2s in ensemble. 
**No major postprocessing** was applied except for tuning scaling factors based on CV scores.
To get the predictions, we aggregated the model outputs at slice level and simply took the maximum value for each patient.

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2Fd6fa2cc524588b85b82906cccb6552bf%2FScreenshot%202023-10-22%20at%206.02.32%20AM.png?generation=1697936329146043&alt=media)

![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2Fb69766140607127e717966085c927bce%2FScreenshot%202023-10-22%20at%206.07.13%20AM.png?generation=1697936371125187&alt=media)

#### **Ensemble:**

Within folds of each models, we are doing slice level ensemble.
For different architectures & cross data models (theo/ours), we did ensemble after the max aggregation. 

#### **Best Ensemble OOF CV**: 0.31x
#### **Best single model 4 fold OOF CV**: 0.326 [Coat lite Medium]

Organ level OOF for single model looks like this:
![](https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F4712534%2F9143afd07ba3069f2b2259b1d8fe80eb%2FScreenshot%202023-10-16%20at%204.05.07%20AM.png?generation=1697413214073394&alt=media)

Thank you. 

3D segmentation code: [notebook link](https://www.kaggle.com/code/haqishen/rsna-2023-1st-place-solution-train-3d-seg/notebook)
