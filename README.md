# DS-NET

Please refer to the paper: Disentangling Style on Dynamic Aligned Poses for Person Identication

Now, the paper is underreviewing.


#--------------------------------------------------

requirements:

tensorflow == 1.14.0

scipy==1.1.0

keras==2.2.0

#--------------------------------------------------
usage:

if you want to train the Auto-Encoder(AE) net from scratch, then just set the line 'flag_train=True' in demo_cnn.py as:

    sess_auto, sample, coding, pred_sample=auto_encoder.AutoEncoderTrain(x_train, x_test, num_epoch=1000, batch_size=256, flag_train=True)
    
Or, if you want to use the trained AE parameters, you should download the model from https://drive.google.com/drive/folders/1-MJjJv8iX2Rkz-hZXaQuQGI6SM-HMp1A?usp=sharing and set 'flag_train=False' as well.
    
You can download the trained model for DS-Net from https://drive.google.com/drive/folders/11VJVmkWJ6y79o1Omkng-YDtpVFcWENwQ?usp=sharing, or you will train the DS-NET from scratch. 

use:

    !python demo_cnn.py

In so doing, I think you should reproduce the same results in the paper, no matter to use the trained model or to train by yourself.
    
