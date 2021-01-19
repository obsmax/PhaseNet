### Changes in this fork : 
- Packaging
    - The program (**phasenet_run.py**) can be called from any location
    - Requirements modified
- Basic graphical display moved to an independant program named **phasenet_disp.py**    
- The prediction mode now provides *absolute* time picks (not only sample indexes)      
- The prediction mode can now run on a SDS data structure for massive datasets
    - no need to group components in a single file  
    - can run on large data structure : preliminary data manipulation no longer required  
- Prediction series for P and S phases can be saved in a hdf5 archive and SDS data structure that mimics the original one.    
    - The overlaps between the 3000 samples windows are better handled  
    - The probability series can be used for further usages like network associations...
- See example usage on anonymous data in demo/demo.sh                   
- **Caution** : mode others than prediction (train/test/...) may have been altered by these changes, use the original fork for these modes. 
Please refer to the original fork in case of doubts. 

### 1. Install using Anaconda
```
# clone this fork to ./PhaseNet (must not exist)
git clone https://github.com/obsmax/PhaseNet.git

# move to the installation directory PhaseNet repo
cd PhaseNet 

conda create --name py36-phasenet python=3.6 --yes
conda activate py36-phasenet

conda install --file requirements.txt --yes
python -m pip install -e .
```


### 2.Demo Data
  
A test mseed SDS data structure is stored in directory: **demo**

### 3.Model

a pretrained model from original fork  
Located in directory: **model/190703-214543**

### 4. Prediction 

#### a) Data format -- mseed with obspy
 
~~~bash
conda activate py36-phasenet
phasenet_run.py \
    --mode pred \
    --model_dir path/to/PhaseNet/PhaseNet/model/190703-214543 \
    --data_dir sds_root_directory \
    --data_list fname.csv \
    --output_dir output_will_be_overwritten \
    --batch_size 20 \
    --input_mseed \
    --save_result
#    --plot_figure  # disabled in this version
~~~

Notes:

1. use --input_mseed (mandatory in this fork)  
2. provide a csv file using the --data_list argument,  
    - header : network,station,location,channel,dataquality,year,julday  
    - one line per day of 3 component data 
        - each line of the csv file must point to exactly 3 files (E,N,Z on a particular day)  
        - use wildcards only for unkown fields and for the component letter (see demo/fname.csv)  
3. The activation thresholds for P&S phases are set to 0.3 as default. Specify **--tp_prob** and **--ts_prob** to change the two thresholds.
4. output  
    - The detected P&S phases are stored to file **picks.csv** inside **--output_dir**
    - if --save-result:  
        - details of the predictions in **output_dir/results/sample_results.hdf5**  
        - a reformed sds archive with predictions **output_dir/results/year/network/...**, channel letters are P or S

 

#### b) Data format -- numpy array
Not recommended on this fork

### 5. Training on new dataset
Not recommended on this fork, use the original one to train the network   
You can then use it with this fork by placing the model in PhaseNet/PhaseNet/model

### Related papers:
- Zhu, W., & Beroza, G. C. (2018). PhaseNet: A Deep-Neural-Network-Based Seismic Arrival Time Picking Method. arXiv preprint arXiv:1803.03211.
- Liu, M., Zhang, M., Zhu, W., Ellsworth, W. L., & Li, H. Rapid Characterization of the July 2019 Ridgecrest, California Earthquake Sequence from Raw Seismic Data using Machine Learning Phase Picker. Geophysical Research Letters, e2019GL086189.
- Park, Y., Mousavi, S. M., Zhu, W., Ellsworth, W. L., & Beroza, G. C. (2020). Machine learning based analysis of the Guy-Greenbrier, Arkansas earthquakes: a tale of two sequences.

