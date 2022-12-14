# Multi-modal Retrieval for Image Synthesis 

This repo includes the code for "A Multi-modal Retrieval Approach for Image
Synthesis based on Diverse Modalities".
- Modalities
  - 1: radiography
  - 2: thickness map
  - 3: synthesized thickness map
- Downstream tasks:
  - 1: KLG prediction
  - 2: Progression prediction

### Installation
```
git clone https://github.com/uncbiag/RetrievalForSynthesis.git
cd RetrievalForSynthesis
conda create -n py37 python=3.7
conda activate py37
pip install -r requirements.txt
```

### Image Synthesis
1. Train image retrieval
   ```
   python train.py --save_name <save folder> --cnn_type <encoder network>
   -bs <batch size>  -e <total epoch>  -g <gpu ids> --region <fc/tc/all>
   -lr1 <modality1 learning rate> -lr2 <modality2 learning rate 2> --lr_update <learning rate update epoch>
   --input_size1 <modality1 input size> --input_size2 <modality2 input size>
   --augmentation --flip --one_per_patient --load_into_memory
   ```  
    
2. Evaluate image retrieval
   ```
   python evaluate.py --resume <checkpoint path> --cnn_type <encoder network>
   -bs <batch size> -g <gpu ids> --region <fc/tc/all>
   --input_size1 <modality1 input size> --input_size2 <modality2 input size> 
   --flip --one_per_patient --load_into_memory
   ```  
   
3. Analysis image synthesis results
   ```
   python retrieve_image.py --resume <checkpoint path> --cnn_type <encoder network>
   -bs <batch size> -g <gpu ids> --region <fc/tc/all> 
   --input_size1 <modality1 input size> --input_size2 <modality2 input size> 
   --flip --load_into_memory
   ```  
    
4. Visualize image synthesis results
   ```
   python visualize.py --resume <checkpoint path> --save_name <save folder> --criterion <id_side_months>
   ``` 

### Downstream Task
1. Train downstream task
   ```
   python pred_train.py --save_name <save folder> --cnn_type <encoder network>
   --resume_fc <fc synthesis result folder> --resume_tc <tc synthesis result folder>
   -bs <batch size> -e <total epoch> -g <gpu_ids>
   -lr2 <learning rate> --lr_update <schedular update epoch> 
   --task <1/2> --modality <2/3> --augmentation --flip --load_into_memory
   ```  
    
2. Evaluate downstream task
   ```
   python pred_eval.py --resume_pred <checkpoint path> --cnn_type <encoder network>
   --resume_fc <fc synthesis result folder> --resume_tc <tc synthesis result folder>
   -bs <batch size> -g <gpu ids> --task <1/2> --modality <2/3> --flip --load_into_memory
   ```  
