from glob import glob
import os

def get_image_list(data_root, split):
    filelist = []
    with open('filelists/{}.txt'.format(split)) as f:
        for line in f:
            line = line.strip()
            if ' ' in line: 
                line = line.split()[0]
                filelist.append(os.path.join(data_root, line))
    return filelist

class HParams:
    def __init__(self):
        self.num_mels=80  # Number of mel-spectrogram channels and local conditioning dimensionality
        self.rescale=True  # Whether to rescale audio prior to preprocessing
        self.rescaling_max=0.9  # Rescaling value

        # Use LWS (https://github.com/Jonathan-LeRoux/lws) for STFT and phase reconstruction
        # It"s preferred to set True to use with https://github.com/r9y9/wavenet_vocoder
        # Does not work if n_ffit is not multiple of hop_size!!
        self.use_lws=False

        self.n_fft=800  # Extra window size is filled with 0 paddings to match this parameter
        self.hop_size=200  # For 16000Hz 200 = 12.5 ms (0.0125 * sample_rate)
        self.win_size=800  # For 16000Hz 800 = 50 ms (If None win_size = n_fft) (0.05 * sample_rate)
        self.sample_rate=16000  # 16000Hz (corresponding to librispeech) (sox --i <filename>)

        self.frame_shift_ms=None  # Can replace hop_size parameter. (Recommended: 12.5)

        # Mel and Linear spectrograms normalization/scaling and clipping
        self.signal_normalization=True
        # Whether to normalize mel spectrograms to some predefined range (following below parameters)
        self.allow_clipping_in_normalization=True  # Only relevant if mel_normalization = True
        self.symmetric_mels=True
        self.max_abs_value=4.
        self.preemphasize=True  # whether to apply filter
        self.preemphasis=0.97  # filter coefficient.

        self.min_level_db=-100
        self.ref_level_db=20
        self.fmin=55
        self.fmax=7600  # To be increased/reduced depending on data.

        self.img_size=96
        self.fps=25

        self.batch_size=16
        self.initial_learning_rate=1e-4
        self.nepochs=200000000000000000  ### ctrl
        self.num_workers=16
        self.checkpoint_interval=3000
        self.eval_interval=3000
        self.save_optimizer_state=True

        self.syncnet_wt=0.0 # is initially zero will be set automatically to 0.03 later. Leads to faster convergence. 
        self.syncnet_batch_size=64
        self.syncnet_lr=1e-4
        self.syncnet_eval_interval=10000
        self.syncnet_checkpoint_interval=10000
        self.disc_wt=0.07
        self.disc_initial_learning_rate=1e-4

hparams = HParams()

def hparams_debug_string():
    values = hparams.values()
    hp = ["  %s: %s" % (name, values[name]) for name in sorted(values) if name != "sentences"]
    return "Hyperparameters:\n" + "\n".join(hp)
