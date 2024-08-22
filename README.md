# Domain-invariant Pretrained(DIP) frontend
The official Domain-invariant Pretrained(DIP) frontend implementation based on fairseq and the pretrained checkpoint as mentioned in System.13.
Our paper link is availiable(https://ieeexplore.ieee.org/document/10640238)

##### Data preparing
To papare the multiple domain data, an example is in dataset/PT_vox2k_full:
1. train_source.tsv, valid_target.tsv are required for source domain synthetic mixture;
2. train_target.tsv, valide_target.tsv are required for target domain real mixture;
The label of mixture is not needed;

#### pretrain model
To pretrain the DIP model, using run.sh;

#### train separator
Our DIP frontend is suitable for both time-domain and frequency domain separator:

1. For frequency domain separator, S3prl toolkit is recommanded;
   
2. For time domain separator, asteroid toolkit is recommanded;

We will update the script for both frontend pretraining, separator training and checkpoint of DIP later. 

### License
The code and models in this repository are licensed under the [GNU General Public License v3](https://www.gnu.org/licenses/gpl-3.0.en.html) for academic and other non-commercial uses. For commercial use, the enquirers will require a license from us or sublicense from a collaborated company if the company decides on an exclusive license for the invention.

Please contact:
- e0125301@u.nus.edu
