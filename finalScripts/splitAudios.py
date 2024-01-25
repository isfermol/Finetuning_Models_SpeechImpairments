import re
import string
from moviepy.editor import *
import os
import  re
import pandas as pd

path = 'media/isabel/EXTERNAL_USB/tfm/EnglishTalkbank/testSpeechIndependent/'
# path2 = '/content/Videos/TCU02a/'
# path2 = '/content/Videos/TCU02a/'
# name = 'ACWT01a.mp4'
audioFolder = path+'Audios/'
for name in os.listdir(path+'videos/'):
    print(name)
    fichero = open(path+'Transcripts/'+name[:-4]+'.cha.txt')
    lineas = fichero.readlines()
    # for linea in lineas:
        # print(linea)

    dialogo = []
    formato = []

    for linea in lineas:
      ans = re.findall(r'^\*PAR.*', linea, re.MULTILINE)
      if ans != []:
        dialogo.append(ans)

    for linea in dialogo:
      linea[0] = linea[0].replace("(","").replace(")","")
      linea[0] = linea[0].replace("\t"," ")
      # print(linea)

      # print(linea[0])
      ans = re.findall(r'(^\*.*\s.)|(\d.*)', linea[0], re.MULTILINE)
      if ans != []:
        formato.append(ans)
        # print(ans[0])
        # if ans[0] != '';
        #  formato.append(ans[0])
        # else:
        #    formato.append(ans[1])

    # for linea in formato:
      # linea[0] = linea[0].replace("\x15","")
    transcripts = []
    times = []
    for linea in formato:
      text = linea[0][0]
      text = text.replace("\x15","")
      text = text.replace("[/]","")
      text = text.replace("*INV:","")
      text = text.replace("*PAR:","")
      text = re.sub(r'\S*(?=\@[a-z])', '', text)
      text = re.sub(r'\b\w+\s*(?=\[:)', '', text)
      

      text = re.sub(r'\@\w', '', text)
      text = re.sub(r'&=[^ ]+', '', text)
      text = text.replace("[+ (\b*\w*\s*)* ]","")
      text= re.sub(r"\[(\+||\*) .*?]","[]", text)
      text= re.sub("xxx","", text)
      text= re.sub("_"," ", text)
      text= re.sub(r"&(\+||\-)\w+","", text)
      text = re.sub(r'\s+(?=\s)', '', text)
      

      if len(linea)>1:
        transcripts.append(text)

        # print(f'linea-> {linea[1]}')
        time = linea[1][1]
        time = time.replace("\x15","")
        mytime = time.split('_')
        times.append(mytime)
      # print(text,',',mytime)

    # print(times)
    # print(transcripts)


    clip = VideoFileClip(path+'videos/'+name[:-4]+".mp4")
    files= audioFolder+name[:-4]

    os.mkdir(files)
    for time in times :
      sb = clip.subclip(int(time[0])/1000,int(time[1])/1000)
      # sb.ipython_display(width = 480,  maxduration = 3000)
      sb.audio.write_audiofile(files+'/audio'+time[0]+'.mp3',verbose=False)


    path = files+'/'
    audios = []
    for archivo in os.listdir(path):
      myfile = path+archivo
      audios.append(myfile)
      # print(myfile)


    # print(f'lenfth audios-> {len(audios)}, len transcriptions-> {len(transcripts[:-1])}')

    testaudios = audios

    testaudios = []
    for archivo in os.listdir(path):
      # myfile = path+archivo
      testaudios.append(archivo)
      # print(archivo)


    # print(testaudios)
    r = re.compile(r"(\d+)")
    testaudios.sort(key=lambda x: int(r.search(x).group(1)))
    # print(testaudios)

    myaudios = []
    for audio in testaudios:
      myaudios.append(path + audio)

    # print(myaudios)

    # transcripts.pop(188)
    # transcripts[357] =transcripts[357]+transcripts[358]
    # transcripts.pop(358)

    # transcripts[42] =transcripts[42]+transcripts[43]
    # transcripts.pop(44)
    # transcripts.pop(189)

    print(len(transcripts), len(myaudios))


    for i in range(0, len(transcripts)):
      # transcripts[i] = re.sub(r'[^a-zA-Z\s.,!?¡¿]', '', transcripts[i])
      transcripts[i] = re.sub(r'[^ a-zA-ZÀ-ÿ\u00f1\u00d1\s.,!?¡¿]', '', transcripts[i])

    trainAudios = myaudios[:-1]
    trainTranscript = transcripts[:-1]

    df_test = pd.DataFrame({"audio": trainAudios[:], "transcription":trainTranscript[:]})
    df_test
    df_test.to_csv('/media/isabel/EXTERNAL_USB/tfm/EnglishTalkbank/csvs/'+name+'.csv')
    path = 'media/isabel/EXTERNAL_USB/tfm/EnglishTalkbank/testSpeechIndependent/'
# path2 = '/content/Videos/TCU02a/'
