import musicnet

root = 'home/hamza/data/musicnet'
window = 16384

#starting download
train_set = musicnet.MusicNet(root=root, train=True, download=True, window=window, mmap=False)#, pitch_shift=5, jitter=.1)