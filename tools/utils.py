import os
import glob
import shutil
import logging
from pathlib import Path
import pandas as pd
import wave
import pysubs2
from threading import Lock, Thread
from queue import Queue
import traceback
from time import sleep
import signal

def clear_folder(folder):
    """
    Удаление всех файлов из директории
    
    Аргументы:
        folder: путь к директории
    """
    files = glob.glob(str(Path(folder) / '*'))
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            print(e)

def delete_folder(folder):
    """
    Удаление директории
    
    Аргументы:
        folder: путь к директории
    """
    try:
        shutil.rmtree(folder, ignore_errors=True)
    except Exception as e:
        print(e)

def prepare_wav(wav):
    """
    Конвертация аудио файла

    Аргументы:
        wav: путь к .WAV файлу аудио

    Результат:
        wav: путь к конвертированному .WAV файлу аудио
    """
    parts = wav.split('.')
    try:
        wave.open(wav, 'r')
    except:
        old_wav = wav + ""
        try:
            wav_file, sr = librosa.load(wav, sr=None, mono=False)
        except:
            return wav
        parts[-1] = "wav"
        wav = '.'.join(parts)
        try:
            soundfile.write(wav, wav_file, sr, format='WAV')
        except:
            return old_wav
        if old_wav != wav:
            os.remove(old_wav)
    return wav

def make_wav_scp(wav, scp):
    """
    Формирование .SCP файла для аудио
    
    Аргументы:
        wav: путь к .WAV файлу аудио
        scp: путь к .SCP файлу с аудио
    """
    with open(scp, 'w') as f:
        wav_file = wave.open(wav, 'r')
        if wav_file.getnchannels() == 1:
            f.write(str(Path(wav).stem) + '.0\t' + 'sox ' + wav + ' -t wav - remix 1 |\n')
        else:
            f.write(str(Path(wav).stem) + '.0\t' + 'sox ' + wav + ' -t wav - remix 1 |\n')
            f.write(str(Path(wav).stem) + '.1\t' + 'sox ' + wav + ' -t wav - remix 2 |\n')          

def make_spk2utt(utt2spk):
    """
    Формирование spk2utt файла
    
    Аргументы:
        utt2spk: путь к файлу сопоставления сегментов и говорящих

    Результат:
        spk2utt: путь к файлу перечисления сегментов для каждого говорящего
    """
    spk2utt = str(Path(utt2spk).parents[0] / 'spk2utt')
    utt2spk_df = pd.read_csv(utt2spk, sep='\t', header=None, names=['utt_id', 'speaker_id'])
    spk2utt_df = utt2spk_df.groupby('speaker_id')['utt_id'].apply(lambda x: ' '.join(x)).reset_index()
    spk2utt_df.to_csv(spk2utt, sep='\t', index=False, header=False)
    return spk2utt

def make_ass(wav, segments, transcriptions, utt2spk, ass):
    """
    Формирование .ASS файла из транскрибаций
    
    Аргументы:
       wav: наименование аудио файла
       segments: путь к файлу описания сегментов
       transcriptions: путь к файлу транскрибации
       utt2spk: путь к файлу сопоставления сегментов и говорящих
       ass: путь к .ASS файлу субтитров
    """
    sub = pysubs2.SSAFile()
    sub.info['Title'] = 'Default Aegisub file'
    sub.info['YCbCr Matrix'] = 'None'
    sub.aegisub_project['Audio File'] = wav
    sub.aegisub_project['Scroll Position'] = 0
    sub.aegisub_project['Active Line'] = 0
    segments_df = pd.read_csv(segments, header=None, sep=' ', names=['utt_id', 'wav', 'start', 'end'])
    transcriptions_df = pd.read_csv(transcriptions, sep='\t', header=None, names=['utt_id', 'text'])
    utt2spk_df = pd.read_csv(utt2spk, sep='\t', header=None, names=['utt_id', 'speaker'])
    events = segments_df.merge(transcriptions_df, how='left', on='utt_id').merge(utt2spk_df, how='left', on='utt_id').fillna('')
    for row in events.values:
        event = pysubs2.SSAEvent(start=pysubs2.make_time(s=float(row[2])), end=pysubs2.make_time(s=float(row[3])), 
                                text=row[4], name=row[5])
        sub.events.append(event)
    sub.sort()
    sub.save(ass, format_='ass')

def create_logger(logger_name, logger_type, logger_level, filename=None):
    """
    Создание логгера
    
    Аргументы:
        logger_name: название логгера
        logger_type: тип логгера
        logger_level: уровень логирования
        filename: название файла лога

    Результат:
        logger: объект логгера        
    """    
    logger = logging.getLogger(logger_name)
    if logger_type == 'file':
        file_handler = logging.FileHandler(filename=filename)
        formatter = logging.Formatter(fmt='%(asctime)s \t %(levelname)s \t %(message)s',
                                            datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(logger_level)
        logger.propagate = False
    elif logger_type == 'stream':
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter(fmt='%(levelname)s: %(message)s (%(asctime)s)')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(logger_level)       
    return logger


class ThreadPool(object):
    def __init__(self, queue_threads, *args, **kwargs):
        self.frozen_pool = kwargs.get('frozen_pool', False)
        self.print_queue = kwargs.get('print_queue', True)
        self.pool_results = []
        self.lock = Lock()
        self.queue_threads = queue_threads
        self.queue = Queue()
        self.threads = []

        for i in range(self.queue_threads):
            t = Thread(target=self.make_pool_call)
            t.daemon = True
            t.start()
            self.threads.append(t)

    def make_pool_call(self):
        while True:
            if self.frozen_pool:
                #print '--> Queue is frozen'
                sleep(1)
                continue

            item = self.queue.get()
            if item is None:
                break

            call = item.get('call', None)
            args = item.get('args', [])
            kwargs = item.get('kwargs', {})
            keep_results = item.get('keep_results', False)

            try:
                result = call(*args, **kwargs)

                if keep_results:
                    self.lock.acquire()
                    self.pool_results.append((item, result))
                    self.lock.release()

            except Exception as e:
                self.lock.acquire()
                print(e)
                traceback.print_exc()
                self.lock.release()
                os.kill(os.getpid(), signal.SIGUSR1)

            self.queue.task_done()

    def finish_pool_queue(self):
        self.frozen_pool = False

        while self.queue.unfinished_tasks > 0:
            if self.print_queue:
                pass
            sleep(5)

        self.queue.join()

        for i in range(self.queue_threads):
            self.queue.put(None)

        for t in self.threads:
            t.join()

        del self.threads[:]

    def get_pool_results(self):
        return self.pool_results

    def clear_pool_results(self):
        del self.pool_results[:]
