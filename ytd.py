
#import os
#os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/df/audio-orchestrator-ffmpeg/bin/ffmpeg"

#from moviepy.config import change_settings
#change_settings({"FFMPEG_BINARY": "/Users/df/audio-orchestrator-ffmpeg/bin/ffmpeg"})

#import ffmpeg


#import os
#os.system('"pip install --upgrade pytube"')


from pytube import YouTube
from pytube import Search



#import moviepy
#import moviepy.editor

#s = Search('amen, brother the winstons')
#len(s.results)
#s.results


#yt = YouTube(str(input("Enter the video link: ")))

#yt.title
#yt.thumbnail_url



my_proxies = None
file_path_glb = None
convert_to_types = ["wav", "mp3"]
prefix = ""
audio_only = True

def progress_func(stream, chunk, bytes_remaining):
    pass
    print("Bytes remaining "+str(bytes_remaining)+"%")
def complete_func(stream, file_path):
    global file_path_glb
    file_path_glb = file_path

    print(F"Download completed for: {str(stream.title)}, in {file_path}, downloaded: {str(stream.filesize)} bytes")
    
    if "mp4" in convert_to_types and len(convert_to_types) == 1:
        print(F"Convertion completed for: {file_path}. Converted to: {convert_to_types}")
        return

    print(F"Converting to {convert_to_types}: {file_path}")
    convert_audio(file_path, convert_to_types)
    print(F"Convertion completed for: {file_path}. Converted to: {convert_to_types}")

def fetch_video(search_string_or_url, proxies=None, quality_max=None, audio_only=True):

    if quality_max is None:
        quality_max = 10000
    
    # om man söker med en url så kommer man få träff på rätt video
    s = Search(search_string_or_url)

    if len(s.results) == 0:
        return None
    else:
        obj = s.results[0]
        url = obj.embed_url

    yt = YouTube(
        url,
        on_progress_callback=progress_func,
        on_complete_callback=complete_func,
        proxies=my_proxies,
        use_oauth=False,
        allow_oauth_cache=True
    )

    streams = yt.streams.filter(only_audio=audio_only, mime_type="video/mp4") 

    if audio_only == True:
        streams = yt.streams.filter(only_audio=audio_only, mime_type="audio/mp4") # denna är det jag behöver, men ffmpeg converter kräven video för att konvertera... 

    # pick highest quality
    # pick highest quality
    all_streams = []
    for s in streams:
        #print(s)
        x = {}
        
        # audio only
        # check if s has property abr
        if "abr" in s.__dict__ and s.abr != None: 
            if s.abr[:-1] == "p":
                # kvalitet är för dålig
                continue

            quality_ = s.abr.replace("kbps","")
            x["quality"] = int(quality_)
            x["itag"] = s.itag
            all_streams.append(x)

    highest_quality = 0
    highest_quality_itag = None
    for x in all_streams:
        if x["quality"] > highest_quality and x["quality"] <= quality_max:
            highest_quality = x["quality"]
            highest_quality_itag = x["itag"]

    stream = yt.streams.get_by_itag(highest_quality_itag)
    stream.download("downloads/")

def fetch_videos(prefix_, search_string_or_urls:list, quality_max=320, audio_formats="mp3", audio_only=True):
    global convert_to_types
    global prefix
    convert_to_types = audio_formats
    prefix = prefix_

    for url in search_string_or_urls:
        fetch_video(url, quality_max=quality_max, audio_only=audio_only)



def convert_audio(file_path, audio_formats=["mp3"], delete_original=True):
    for audio_format in audio_formats:
        ffmpeg.input(file_path).output(str(file_path).replace(".mp4", "") + F".{audio_format}", f=audio_format).run(cmd=r"/Users/df/audio-orchestrator-ffmpeg/bin/ffmpeg", overwrite_output=True)

    if delete_original == True:
        os.remove(file_path)

song_list = ["amen, brother the winstons"
    , "lyn collins think (about it)"
    , "funky drummer james brown"
    , "do the do the deuce"
    , "seven b lover & a friend"
    , "blow fly on tv, ed sullivan show" 
    , "blow fly on tv, sesame street" 
    , "blow fly on tv, batman" 
    , "blow fly on tv, signed & sealed" 
    , "blow fly on tv, mammy told me" 
    , "blow fly on tv, the love boat"]

song_list = ["pump up the jam technotronics", "pump up the volume"]
song_list = ["theoz theori"]

song_list = ["l31RXiVSI9s"]
song_list = ["lyn collins think (about it)"]

song_list = ["Theoz - Christmas Song (Official Lyric Video)", "Theoz - Het (Official Video)", "Theoz - Som du vill", "theoz theori", "theoz hooked on a feeling", "theoz julmusiken"]

song_list = ["YkLayQGiHPU"] #lloyd price hooked on a feeling
song_list = ["7Z3uhIRkBNw"] #björn shiffs hook on a feeling 
song_list = ["TazHNpt6OTo", "V9AbeALNVkk", "FTQbiNvZqaY", "I_izvAbhExY", "09839DpTctU"] #pina collada song, toto, twister sister
song_list = ["4xmckWVPRaI"] #björn shiffs hook on a feeling 
song_list = ["fJ9rUzIMcZQ", "ZhIsAZO5gl0", "JmcA9LIIXWw", "XfR9iY5y94s", "kd9TlGDZGkI", "OMOGaugKpzs", "wMsazR6Tnf8"] #björn shiffs hook on a feeling 

song_list = ["g3wz3Lg3Tho"]

song_list = ["qTu8QCL7soQ"] # Banbarra

song_list = ["DdW_Q0fxwM"] # where only buggin
song_list = ["SdUl9cYZjYc"] # random speach for wisper speach to text.  

# BLUEGRASS TRIBUTE TO THE OFFSPRING
song_list = ["lJNo_frQSA8", "kNyvqmO4Sdg", "ZO9YuC7eBrk", "v1cQTnL9Ys8" , "91_teqNqlcQ", "gF7z3eznoWA", "syHj0l3WpGw", "CAinkrM53Co", "36Jt_P4muCo", "VMfUO8p-N6s", "lJNo_frQSA8"]
song_list = ["91_teqNqlcQ", "gF7z3eznoWA", "syHj0l3WpGw", "CAinkrM53Co", "36Jt_P4muCo", "VMfUO8p-N6s", "lJNo_frQSA8"]
song_list = ["yOLSMIB1JCw"]
song_list = ["cMr31qCnx5c"] # MJ’s Thriller Frog Bass … Found!
song_list = ["qfcRlQ4swrE"] # MJ’s80s drums - 02 Catch Me I'm Falling Melô do Poder - ( Lp Furacão 2000) - 1989
song_list = ["qF6wKq6l9w4"] # MJ’s80s drums - Its automatic 2.05, 4.25 808 - ( Lp Furacão 2000) - 1989

song_list = ["AxT_IJ-ue54"] # powerhouse 
song_list = ["kEjNY5eW9Gg"] #Critical Beatdown

# check if run though a terminal and has arguments


import sys

# write instrucion if no arguments
#print("Skriv URL eller ID på youtube video ")

if len(sys.argv) > 1:
    song_list = sys.argv[1:]

if len(sys.argv) == 1:
    print("Skriv URL eller ID för youtube video att ladda ner. Example: python youtube_downloader2.py qF6wKq6l9w4")
    sys.exit()

#fetch_audios("", song_list, quality_max=9999, audio_formats=["mp3", "wav"], audio_only=True)
fetch_videos("", song_list, quality_max=9999, audio_formats=["mp4"], audio_only=False)


# yt = YouTube("https://www.youtube.com/watch?v=7Z3uhIRkBNw"
    # , on_progress_callback=progress_func
    # , on_complete_callback=complete_func
    # , proxies=my_proxies
    # , use_oauth=False
    # , allow_oauth_cache=True
    # )









