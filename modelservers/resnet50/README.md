# Create a pretrained ResNet50 model server and run a load test

#### 1. Create the model

##### Using a Spell Run
1. Create a run with downloads the model `spell run python make_resnet50.py`
2. Create a Spell model from the H5 model file using `spell model create --file modelservers/resnet50/model.h5:model.h5 resnet50 runs/<RUN_ID>`

##### Uploading the model
1. Locally run `python make_resnet50.py`
2. Upload the resulting `model.h5` model using `spell upload --name resnet50.h5 ./model.h5`
3. Create a Spell model from the uploaded H5 model file using `spell model create --file resnet50.h5:model.h5 resnet50 uploads/resnet50.h5`

#### 2. Start the model servers
Start the unbatched predictor with 
```shell
$ spell server serve \
  --pip 'h5py<3' \
  --name resnet50 \
  --env MODEL_PATH=/model/model.h5 \
  --node-group <YOUR_NODE_GROUP_WITH_GPU> \
  --gpu-limit 1 \
  --no-open \
  --classname Predictor \
  resnet50:v1 predictor.py
```
And start the batched predictor with
```shell
$ spell server serve \
  --pip 'h5py<3' \
  --name resnet50 \
  --env MODEL_PATH=/model/model.h5 \
  --node-group <YOUR_NODE_GROUP_WITH_GPU> \
  --gpu-limit 1 \
  --classname BatchPredictor \
  --request-timeout 25 \
  resnet50:v1 predictor.py
```

#### 3. Run the load test
Run the load test on a Spell run to get a more consistent network throughput. 
```shell
$ spell run \
  --pip aiohttp \
  --pip dataclasses \
  --label loadtest \
  --description "Load test on ResNet50 model server" \
  -- \
    python loadtest.py \
    --hold-seconds 10 \
    --procs 10 \
    --url <URL_TO_YOUR_MODEL_SERVER> \
    --name unbatched \
    --out-dir ./loadtest \
    --rates 10:1000:10 \
    --latency-limit 1000 \
    --img-path cat.jpeg
```
And 
```shell
$ spell run \
  --pip aiohttp \
  --pip dataclasses \
  --label loadtest \
  --description "Load test on batched ResNet50 model server with request timeout of 25ms" \
  -- \
    python loadtest.py \
    --hold-seconds 10 \
    --procs 10 \
    --url <URL_TO_YOUR_BATCHED_MODEL_SERVER> \
    --name batched-t25 \
    --out-dir ./loadtest \
    --rates 10:1000:10 \
    --latency-limit 1000 \
    --img-path cat.jpeg
```

#### 4. Gather the results

```shell
$ mkdir ./loadtest-results
$ spell cp runs/<UNBATCHED_RUN_ID>/modelservers/resnet50/loadtest loadtest
$ spell cp runs/<BATCHED_RUN_ID>/modelservers/resnet50/loadtest loadtest
```

#### 5. Visualize the results
##### Locally
Start a Jupyter notebook and open the `loadtest.ipynb` notebook

##### On Spell

```shell
$ spell jupyter \
  --lab \
  --mount uploads/loadtest.ipynb
  --mount runs/<UNBATCHED_RUN_ID>/modelservers/resnet50/loadtest/unbatched/consolidated.csv:unbatched.csv \
  --mount runs/<BATCHED_RUN_ID>/modelservers/resnet50/loadtest/batched-t25/consolidated.csv:batched-t25.csv \
  loadtest
```
Inside the workspace, open the `modelservers/resnet50/loadtest.ipynb` file and modify the notebook to visualize your results.