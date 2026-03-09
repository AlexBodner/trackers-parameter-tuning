<h1> Extracted YOLOX detections for the <u>DanceTrack</u> dataset </h1>

Related notebooks:
<ul> 
<li>ByteTrack runs on DanceTrack sequences: <a href="https://colab.research.google.com/drive/1ygRZf0aMCBka6APleltumKIt0BOIk2gI">link</a></i>
<li>ByteTrack evaluation on DanceTrack sequences: <a href="https://colab.research.google.com/drive/1W9Qg7gTyIwksp1SDkyUUhOmLF9sxanVo">link</a></i>
</ul>

YOLOX detections as used in <a href="https://github.com/FoundationVision/ByteTrack">ByteTrack</a> and in other methods building on top of it, such as <a href="https://github.com/tstanczyk95/McByte">McByte</a>, <a href="https://github.com/noahcao/OC_SORT">OC-SORT</a>, etc.

<h2> Format </h2>

For each sequence, the detection file has a name of `sequence_name.txt` and is palced under the relevant split directory (train, val, test).
<br/>
The format of each record in the detection files is:
<br/>
`frame_no,x1,y1,x2,y2,det_conf`
<br/><br/>
E.g. first records of the file could be:
<br/>

`1,781.6500,551.8125,881.5500,842.4000,0.93`<br/>
`1,688.1625,530.2125,788.4000,837.0000,0.92`<br/>
`1,1119.1500,574.0875,1185.3000,822.1500,0.90`<br/>
`1,1070.5500,608.8500,1189.3501,826.2000,0.90`<br/>
`1,926.1000,562.9500,997.6500,824.8500,0.90`<br/>
`1,811.3500,647.3250,951.7501,859.2750,0.88`<br/>
`1,962.5500,635.8500,1073.2500,857.2500,0.85`<br/>
`1,649.0125,560.2500,776.2500,831.6000,0.82`<br/>
`1,1013.1750,634.5000,1119.8251,838.3500,0.77`<br/>
`1,1021.9500,568.3500,1096.2001,815.4000,0.67`<br/>
`1,924.7500,642.2625,1004.4000,830.9250,0.52`<br/>
`1,924.7500,640.2375,1047.6000,830.2500,0.33`<br/>
(All obtained detections at frame 1.)<br/>


<h2> Parameters and settings used </h2>
These detections were extracted using YOLOX mounted within ByteTrack. Adjusted version of ByteTrack's tools/demo_track.py was run, where detections returned by the predictor were collected as <a href="https://github.com/FoundationVision/ByteTrack/blob/main/tools/demo_track.py#L189">here</a>. No tracking was performed at that time, only the detection collection.
<br/><br/>

<h3>Hardware and packages</h3>
The runs were performed with the following components:
<ul>
<li>torch '2.9.1+cu128'</li>
<li>compiled (python3 setup.py develop) with gcc 10.4.0</li>
<li>run on a signle H100 GPU.</li>
</ul>

<br/>
<h3>Parameters</h3>
For the consistency reasons with the original ByteTrack settings, the following parameters were set and the following models were used.
<br/><br/>
Parameters were placed after the python command, e.g.<br/>

`python3 tools/demo_track.py param1 --param2 value2 --param3 value3`<br/>

All 3 datasets (MOT17, DanceTrack, SportsMOT):<br/>
`image --fp16 --conf 0.01 --nms 0.7 --path /path/to/a/single/sequence`<br/>

Then additionally for: <br/>
DanceTrack train, val and test: `-f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_dancetrack.pth.tar` <br/> <br/>
<br/>
E.g. for DanceTrack test set, a single sequence detection extracting command would be:<br/>

`python3 tools/demo_track__only_save_dets.py image -f exps/example/mot/yolox_x_mix_det.py -c pretrained/bytetrack_x_dancetrack.pth.tar --fp16 --conf 0.01 --nms 0.7 --path /my_dir/datasets/dancetrack/test/dancetrack0003/img1`

<h2> Pre-trained models </h2>
The pre-trained detection models are available here:<br/>
<a href="https://huggingface.co/noahcao/dancetrack_models/tree/main/bytetrack_models">DanceTrack</a> - bytetrack_model.pth.tar<br/>

In case of the pre-trained model unavailability, their backups are available <a href="https://drive.google.com/drive/folders/1yzzJk9dpJUY3lIHdkkFyGtKL2F-FenN6">here</a>.<br/> 
