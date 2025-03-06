# Domain-invariant Pretrained(DIP) frontend
The Domain-invariant Pretrained(DIP) frontend implementation based on fairseq and the pretrained checkpoint as mentioned in System.13.  
Our paper link is available (https://ieeexplore.ieee.org/document/10640238)  
The checkpoint of the pretrained frontend is available (https://drive.google.com/file/d/15_YZAF1MJs4VybuYM2FSsLQau_I0qir-/view?usp=drive_link)  

##### Data preparing
To prepare the multiple domain data, an example is in dataset/PT_vox2k_full:
1. train_source.tsv, valid_target.tsv are required for source domain synthetic mixture;
2. train_target.tsv, valide_target.tsv are required for target domain real mixture;
The label of the mixture is not needed;

#### Pretrain model
To pretrain the DIP model, using run.sh;

#### Train separator
Our DIP frontend is suitable for both time-domain and frequency-domain separators:

1. For frequency domain separator, the S3prl (https://github.com/s3prl/s3prl) toolkit is recommended;  
   Please replace the original s3prl/upstream/wav2vec2/expert.py with our version in separation/expert.py.The test results on Libri2Mix should be about 11dB:  
   ![S3PRL](img/S3PRL_Libri2Mix_DIP.PNG "S3PRL_DIP_Libri2Mix")  
   For test results on the Vox2Mix test set, please refer to the paper.  
   
3. For the time domain separator, the asteroid toolkit (https://github.com/asteroid-team/asteroid) is recommended;  
   For test results on Vox2Mix, LRS2Mix, and Real-M test sets, please refer to the paper.   

We have not checked the correctness of all scripts. Please do neccessary modification for your pretraining and training process. We will update the script for both frontend pretraining and separator training later.     

### License
The code and models in this repository are licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html) for academic and other non-commercial uses. For commercial use, the enquirers will require a license from us or sublicense from a collaborated company if the company decides on an exclusive license for the invention.

Please contact:
- e0125301@u.nus.edu
