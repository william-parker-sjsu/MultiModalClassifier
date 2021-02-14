import pytube #pip install pytube
from pytube import YouTube #https://github.com/nficano/pytube
import os  
import subprocess
import sys
from pytube import Playlist
p = Playlist('https://www.youtube.com/playlist?list=PL_8o25--oHQMwiYMm1BSjDw22NRV_uySC')
for videostreams in p.videos:
  video = videostreams.streams.filter(file_extension='mp4', res="720p", adaptive=True).first()
  if video is None:
    videostreams.streams
    continue
  title=video.title
  simpletitle=''.join(e for e in title if e.isalnum())
  videofilename=f'{simpletitle}.mp4'
  print(videofilename)
  video.download('videos/', simpletitle)#output_path: 'videos/', filename: videofilename)
  #!mkdir TmpImages
  #!ffmpeg -i ./videos/$videofilename -vf fps=1 ./TmpImages/$simpletitle-%d.jpg
#   ffmpegcmdstr='ffmpeg -i '+f'./videos/{simpletitle}.mp4 '+ '-vf fps=1 ./TmpImages/'+simpletitle+'-%d.jpg'
#   result = subprocess.call(ffmpegcmdstr, shell=True)