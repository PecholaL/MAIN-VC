# MAIN-VC

## Demo Page
The demo page for MAIN-VC can be found [👉🏻here](https://pecholal.github.io/MAIN-VC-demo/)

## Preproces
Directory ```data``` is for data preprocess, including getting mel-spectrograms and making dataset/data loader. After ensuring the dataset location and correctly setting the paths, use this command to process the data:
```
./preprocess.sh
```

## Train Model
```
./train.sh
```

## Inference
Once the pretrained vocoder is placed in the specified location, set the source speech, target speech, and output path for the results in ```VCinference.sh```. Then use the command to perform inference.
```
./VCinference.sh
```

## Acknowledgements:
* The encoders and decoder is modified from [AdaIN-VC](https://github.com/jjery2243542/adaptive_voice_conversion);
* The mutual information module is modified from [CLUB](https://github.com/Linear95/CLUB);
