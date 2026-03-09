#pip install SoccerNet --upgrade

import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="soccernet_tracking/train")
mySoccerNetDownloader.downloadDataTask(task="tracking", split=["train"])

