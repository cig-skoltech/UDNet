
The code in this package implements grayscale and color image denoising as described in the paper:

  Stamatis Lefkimmiatis  
  Universal Denoising Networks : A Novel CNN Architecture for Image Denoising  
  IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Salt Lake City, UT, June 2018.  

Please cite the paper if you are using this code in your research.  
Please see the file LICENSE.txt for the license governing this code.  


Overview  

The function UDNET_DENOISE_DEMO demonstrates grayscale and color image denoising with the trained models from the paper, which can all be found in the folder networks-inference.  The paper and supplementary material are provided in the folder "paper".  

The script BSDSValidation located in the folder networks-inference can be used to reproduce the results reported in the paper for each one of the  trained models for the validation set BSDS68, which is extracted from the BSDS500 dataset.  

The folder matlab/custom_layers contains all the CNN layers that are used to build the local and non-local networks described in the CVPR paper, while the folder matlab/+misc includes some miscellaneous functions. The folder matlab/custom_mex includes cpu and gpu mex files used to define some of the layers for the non-local networks.   

Training of the networks  UNET and UNLNET for grayscale and color images can be accomplished   
using the scripts provided in the folder networks-training.   

NOTE : In order to add the package into Matlab's path you need to first execute the script vl_setupnn located in the folder matlab/vl_layers/

Dependencies  

The provided code has dependencies on the MatConvnet toolbox. The necessary functions are included in the folders matlab/vl_layers, matlab/mex, matlab/src and matlab/compatibility.

Contact  

If you have questions, problems with the code, or found a bug, please let us know.  
Contact Stamatis Lefkimmiatis at s.lefkimmiatis@skoltech.ru
