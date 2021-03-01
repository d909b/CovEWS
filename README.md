# COVID-19 Early Warning System (CovEWS)

![Code Coverage](https://img.shields.io/badge/Python-3.7-blue)

The COVID-19 Early Warning System (CovEWS) is a real-time early warning system for COVID-19 related mortality risk. CovEWS stratifies patients into risk percentiles using a number of personal risk factors.

This repository contains the source code for the neural time-varying Cox model used by CovEWS to make its predictions, and a synthetic mock dataset that can be used to run the source code. A detailed description of CovEWS was published [Nature Communications](https://www.nature.com/articles/s41467-020-20816-7)

## Install

The latest release can be installed directly from Github (ideally into a virtual environment):

```bash
$ pip install git+https://github.com/d909b/covews.git
```

Installation should finish within minutes on standard hardware. This software has been tested with Python 3.7.

## Use

Training of a neural time-varying Cox model, based on the synthetic mock patient data in `/path/to/mock_patients.pickle` can be initiated from the command line as follows:

```bash
$ python3 /path/to/covews/apps/main.py 
    --dataset_cache="/path/to/mock_patients.pickle"
    --dataset="covews_mortality"
    --loader=pickle
    --output_directory=/path/to/output_dir
    --do_evaluate
    --do_train
    --method=NonlinearTimeVaryingCox
    --l2_weight=0.1
    --learning_rate=0.001
    --dropout=0.2
    --batch_size=100
    --num_units=128
    --num_layers=2
    --num_epochs=50
    --do_calibrate
```

Training should finish within minutes on standard hardware. After conclusion of the training, the outputs of the training process are written to `/path/to/output_dir` - including output predictions, a model binary, preprocessors, and training loss curves. In addition, relevant performance metric for different prediction horizons are written to stdout.

## Cite

Please consider citing, if you reference or use our methodology, code or results in your work:

    @article{schwab2020covews,
        title={{Real-time Prediction of COVID-19 related Mortality using Electronic Health Records}},
        author={Patrick Schwab and Arash Mehrjou and Sonali Parbhoo and Leo Anthony Celi and Jürgen Hetzel and Markus Hofer and Bernhard Schölkopf and Stefan Bauer},
        year={2021},
        volume={12},
        pages={1058},
        journal={Nature Communications}
    }

### License

[MIT License](LICENSE.txt)

### Data Availability

Accredited users may license the TriNetX COVID-19 research and Optum de-identified COVID-19 electronic health record databases used in our report at TriNetX and Optum, respectively. The dataset contained within this repository consists of synthetic mock patients only, and has no resemblance of real patient data.

### Authors

Patrick Schwab, Arash Mehrjou, Sonali Parbhoo, Leo Anthony Celi, Jürgen Hetzel, Markus Hofer, Bernhard Schölkopf, Stefan Bauer

### Acknowledgements

SP is supported by the Swiss National Science Foundation under P2BSP2_184359. LAC is funded by the National Institute of Health through NIBIB R01 EB017205. BS is a member of the excellence cluster “Machine Learning in the Sciences” funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) under Germany’s Excellence Strategy – EXC number 2064/1 – Project number 390727645. We thank Annika Buchholz for helpful discussions. 
