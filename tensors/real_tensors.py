import numpy as np
import os
from os.path import dirname, join
from scipy.io import loadmat
from .utils import download_unzip_data, load_images_from_folder


def graph_state_5_party(tenpy):
    zero = np.asarray([1., 0.])
    one = np.asarray([0., 1.])
    plus = 1. / np.sqrt(2) * np.asarray([1., 1.])
    minus = 1. / np.sqrt(2) * np.asarray([1., -1.])

    out1 = 1. / np.sqrt(2) * np.einsum("a,b,c,d,e->abcde", zero, plus, zero,
                                       minus, one)
    out2 = 1. / 2 * np.einsum("a,b,c,d,e->abcde", plus, zero, minus, one, plus)
    out3 = 1. / 2 * np.einsum("a,b,c,d,e->abcde", minus, one, plus, one, plus)
    out4 = 1. / 2 * np.einsum("a,b,c,d,e->abcde", plus, zero, plus, zero,
                              minus)
    out5 = 1. / np.sqrt(2) * np.einsum("a,b,c,d,e->abcde", zero, minus, one,
                                       plus, one)
    out6 = 1. / 2 * np.einsum("a,b,c,d,e->abcde", minus, one, minus, zero,
                              minus)
    out = out1 + out2 + out3 + out4 + out5 + out6
    tensor = out.transpose((0, 1, 3, 2, 4)).reshape((2, 4, 4))
    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(tensor)
    return tensor


def amino_acids(tenpy):
    """
    Data: 
        This data set consists of five simple laboratory-made samples. 
        Each sample contains different amounts of tyrosine, tryptophan and phenylalanine 
        dissolved in phosphate buffered water. 
        The samples were measured by fluorescence 
        (excitation 250-300 nm, emission 250-450 nm, 1 nm intervals) 
        on a PE LS50B spectrofluorometer with excitation slit-width of 2.5 nm, 
        an emission slit-width of 10 nm and a scan-speed of 1500 nm/s. 
        The array to be decomposed is hence 5 x 51 x 201. 
        Ideally these data should be describable with three PARAFAC components. 
        This is so because each individual amino acid gives a rank-one contribution to the data.
    References: 
        http://www.models.life.ku.dk/Amino_Acid_fluo
        Bro, R, PARAFAC: Tutorial and applications, 
        Chemometrics and Intelligent Laboratory Systems, 1997, 38, 149-171

    """
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'data')

    urls = ['http://models.life.ku.dk/sites/default/files/Amino_Acid_fluo.zip']
    zip_names = ['Amino_Acid_fluo.zip']
    tensor_name = 'amino.mat'

    download_unzip_data(urls, zip_names, data_dir)
    tensor = loadmat(join(data_dir, tensor_name))['X'].reshape((5, 61, 201))

    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(tensor)
    return tensor


def coil_100(tenpy):
    """
    Columbia University Image Library (COIL-100)
    References:
        http://www.cs.columbia.edu/CAVE/software/softlib/coil-100.php
        Columbia Object Image Library (COIL-100), S. A. Nene, S. K. Nayar and H. Murase, 
        Technical Report CUCS-006-96, February 1996.

    """
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'saved-tensors')

    def create_bin():
        urls = [
            'http://www.cs.columbia.edu/CAVE/databases/SLAM_coil-20_coil-100/coil-100/coil-100.zip'
        ]
        zip_names = ['coil-100.zip']
        file_name = 'coil-100/'
        download_unzip_data(urls, zip_names, data_dir)

        coil_folder = join(data_dir, file_name)
        nonimage_names = ['convertGroupppm2png.pl', 'convertGroupppm2png.pl~']
        for file in nonimage_names:
            nonimage_path = join(coil_folder, file)
            if os.path.isfile(nonimage_path):
                os.remove(nonimage_path)

        pixel = load_images_from_folder(coil_folder)
        pixel_out = np.reshape(pixel, (7200, 128, 128, 3)).astype(float)

        output_file = open(join(data_dir, 'coil-100.bin'), 'wb')
        print("Print out pixels ......")
        pixel_out.tofile(output_file)
        output_file.close()

    if not os.path.isfile(join(data_dir, 'coil-100.bin')):
        create_bin()
    pixels = np.fromfile(join(data_dir, 'coil-100.bin'), dtype=float).reshape(
        (7200, 128, 128, 3))

    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(pixels)
    return pixels[:, :, :, 0]#pixels[:, :, :, :]


def time_lapse_images(tenpy):
    """
    Time-Lapse Hyperspectral Radiance Images of Natural Scenes 2015
    Datasets are under the CCBY license (http://creativecommons.org/licenses/by/4.0/).
    References:
        https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/Time-Lapse_HSIs_2015.html
        Foster, D.H., Amano, K., & Nascimento, S.M.C. (2016). Time-lapse ratios of cone excitations 
        in natural scenes. Vision Research, 120, 45-60.doi.org/10.1016/j.visres.2015.03.012

    """
    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'saved-tensors')

    def create_bin():
        urls = [
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1140.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1240.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1345.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1441.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1600.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1637.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1745.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1845.zip',
            'https://personalpages.manchester.ac.uk/staff/d.h.foster/Time-Lapse_HSIs/nogueiro/nogueiro_1941.zip'
        ]
        zip_names = [
            'nogueiro_1140.zip', 'nogueiro_1240.zip', 'nogueiro_1345.zip',
            'nogueiro_1441.zip', 'nogueiro_1600.zip', 'nogueiro_1637.zip',
            'nogueiro_1745.zip', 'nogueiro_1845.zip', 'nogueiro_1941.zip'
        ]
        download_unzip_data(urls, zip_names, data_dir)

        x = []
        print("Loading 1st dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1140.mat')['hsi'])
        print("Loading 2nd dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1240.mat')['hsi'])
        print("Loading 3rd dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1345.mat')['hsi'])
        print("Loading 4th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1441.mat')['hsi'])
        print("Loading 5th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1600.mat')['hsi'])
        print("Loading 6th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1637.mat')['hsi'])
        print("Loading 7th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1745.mat')['hsi'])
        print("Loading 8th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1845.mat')['hsi'])
        print("Loading 9th dataset ......")
        x.append(loadmat(data_dir + '/nogueiro_1941.mat')['hsi'])
        x = np.asarray(x).astype(float)
        print(x.shape)

        output_file = open(join(data_dir, 'time-lapse.bin'), 'wb')
        print("Print out data ......")
        x.tofile(output_file)
        output_file.close()

    if not os.path.isfile(join(data_dir, 'time-lapse.bin')):
        create_bin()
    pixels = np.fromfile(join(data_dir, 'time-lapse.bin'),
                         dtype=float).reshape((9, 1024, 1344, 33))

    if tenpy.name() == 'ctf':
        return tenpy.from_nparray(pixels)
    return pixels[0, :, :, :]


def get_scf_tensor(n=15):

    from pyscf import gto, scf

    # mol = gto.Mole(basis='ccpvdz')
    mol = gto.Mole(basis='sto-3g')
    mol.atom = [['O', (0, 0, 0)], ['H', (0, 1, 0)], ['H', (0, 0, 1)]]
    mol.atom.extend([['O', (i, 0, 0)] for i in range(1, n)])
    mol.atom.extend([['H', (i, 1, 0)] for i in range(1, n)])
    mol.atom.extend([['H', (i, 0, 1)] for i in range(1, n)])

    # mol = gto.Mole(basis='def2-tzvp')
    # n = 20
    # mol.atom = [['H',(0, 0, 0)]]
    # mol.atom.extend([['H', (i, i, i)] for i in range(1,n)])
    print(mol.atom)
    mf = scf.RHF(mol).density_fit().run()
    T_sym = mf.with_df._cderi
    print(T_sym.shape)
    NN1 = T_sym.shape[1]
    N = int(np.floor(np.sqrt(NN1 * 2)))
    print(N, N * (N + 1) // 2, NN1)
    T = np.zeros((T_sym.shape[0], N, N))
    print(T.shape)
    for i in range(T.shape[0]):
        T[i][np.triu_indices(N)] = T_sym[i][:]
        T[i] += T[i].T
        T[i] -= np.diag(np.diagonal(T[i]) / 2.)
    return T


def get_bert_embedding_tensor(tenpy):

    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'saved-tensors/')
    try:
        os.stat(data_dir + 'BERT-word-embedding.npy')
    except:
        os.system('bash saved-tensors/download.sh embedding')

    tensor = tenpy.load_tensor_from_file(data_dir + 'BERT-word-embedding.npy')
    assert (tensor.shape == (30522, 768))
    return tenpy.reshape(tensor[:30276, :], (174, 174, 768))


def get_bert_weights_tensor(tenpy):

    parent_dir = join(dirname(__file__), os.pardir)
    data_dir = join(parent_dir, 'saved-tensors/')
    try:
        os.stat(data_dir + 'bert_param')
    except:
        os.system('cd saved-tensors; bash download.sh bert-param')
        os.system('cd saved-tensors; tar -xzf bert_param.tar.gz')

    # tensor = tenpy.load_tensor_from_file(data_dir+'bert_param/querys.npy')
    # tensor = tenpy.load_tensor_from_file(data_dir+'bert_param/keys.npy')
    # tensor = tenpy.load_tensor_from_file(data_dir+'bert_param/values.npy')
    # tensor = tenpy.load_tensor_from_file(data_dir+'bert_param/outputs.npy')
    tensor = tenpy.load_tensor_from_file(data_dir +
                                         'bert_param/intermediate_denses.npy')
    # tensor = tenpy.load_tensor_from_file(data_dir+'bert_param/output_denses.npy')
    # assert(tensor.shape == (12, 768, 768))
    assert (tensor.shape == (12, 3072, 768))
    # assert(tensor.shape == (12, 768, 3072))
    return tensor  #tenpy.reshape(tensor[0,:,:], (1,3072,768))
