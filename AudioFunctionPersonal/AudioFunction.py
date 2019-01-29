import pyaudio
import wave
from AudioFunctionPersonal import Params


def playaudiowav(WAVE_INPUT_FILENAME):
    '''if len(sys.argv) < 2:
        print("Plays a wave file.\n\nUsage: %s filename.wav" % sys.argv[0])
        sys.exit(-1)
    '''
    wf = wave.open(WAVE_INPUT_FILENAME, 'rb')
    p = pyaudio.PyAudio()
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    data = wf.readframes(Params.CHUNK)
    while data != '':
        stream.write(data)
        data = wf.readframes(Params.CHUNK)
    stream.stop_stream()
    stream.close()
    p.terminate()

def recordaudiowav(WAVE_OUTPUT_FILENAME):
    p = pyaudio.PyAudio()
    stream = p.open(format=Params.FORMAT,
                    channels=Params.CHANNELS,
                    rate=Params.RATE,
                    input=True,
                    frames_per_buffer=Params.CHUNK)
    print("* recording")
    frames = []
    for i in range(0, int(Params.RATE / Params.CHUNK * Params.RECORD_SECONDS)):
        data = stream.read(Params.CHUNK)
        frames.append(data)
    print("* done recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(Params.CHANNELS)
    wf.setsampwidth(p.get_sample_size(Params.FORMAT))
    wf.setframerate(Params.RATE)
    wf.writeframes(b''.join(frames))
    wf.close()