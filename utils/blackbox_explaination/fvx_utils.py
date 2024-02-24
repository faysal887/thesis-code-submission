import numpy as np
import fnmatch
from os import listdir
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import time, pdb
# from arcface import ArcFace
from deepface import DeepFace

# import face_recognition
from skimage.feature import local_binary_pattern
import itertools
from scipy import ndimage
import torch
import PIL
from facenet_pytorch import InceptionResnetV1
import random
import pickle
import skimage.io 
import skimage.segmentation
import copy
import sklearn
import sklearn.metrics
from sklearn.linear_model import LinearRegression


face_model=None
model_name=None
detector_name=None

# MODEL_NAME='vggface2'
# MODEL_NAME='casia_webface'
# MODEL_NAME='lbp'

# MODEL_NAME='dlib'
# MODEL_NAME='arcface'


def set_face_recognition_model(my_model_name, my_detector_name):
  global face_model
  global model_name
  global detector_name

  model_name = my_model_name
  detector_name = my_detector_name

  '''
  if MODEL_NAME=='arcface':
    # arcface    = ArcFace.ArcFace()
    model_name = 'arcface'
    # face_model = arcface

  elif MODEL_NAME=='dlib':
    dlib = face_recognition
    model_name = 'dlib'
    face_model = dlib

  elif MODEL_NAME=='lbp':
    model_name = 'lbp'
    face_model  = None
  '''

  """
  if MODEL_NAME=='vggface2':
    vggface2      = InceptionResnetV1(pretrained='vggface2').eval()
    model_name = 'vggface2'
    face_model = vggface2

  elif MODEL_NAME=='casia-webface':
    casia_webface = InceptionResnetV1(pretrained='casia-webface').eval()
    model_name = 'casia_webface'
    face_model = casia_webface
  """

  # if MODEL_NAME=="VGG-Face":
  #   model_name = 'VGG-Face'
    
  # elif MODEL_NAME=="Facenet":
  #   model_name = 'Facenet'
  
  # elif MODEL_NAME=="Facenet512":
  #   model_name = 'Facenet512'
  
  # elif MODEL_NAME=="OpenFace":
  #   model_name = 'OpenFace'
  
  # elif MODEL_NAME=="DeepFace":
  #   model_name = 'DeepFace'
  
  # elif MODEL_NAME=="DeepID":
  #   model_name = 'DeepID'
  
  # elif MODEL_NAME=="ArcFace":
  #   model_name = 'ArcFace'
  
  # elif MODEL_NAME=="Dlib":
  #   model_name = 'Dlib'
  
  # elif MODEL_NAME=="SFace":
  #   model_name = 'SFace'

  # return face_model

## Input/Output/Misc
def howis(X):
  print('size = '+str(X.shape))
  print('min  = '+str(X.min()))
  print('max  = '+str(X.max()))


def randomints(i1,i2,n):
  randomlist = []
  i2 = i2-1
  for i in range(n):
    x = random.randint(i1,i2)
    randomlist.append(x)
  return randomlist


def pkl_save(fpath,X):
    output = open(fpath+'.pkl', 'wb')
    pickle.dump(X, output)
    output.close()


def pkl_load(fpath):
    pkl_file = open(fpath+'.pkl', 'rb')
    X = pickle.load(pkl_file)
    pkl_file.close()
    return X


def num2fixstr(x,d):
    st = '%0*d' % (d,x)
    return st


def str2(x):
    return "{:.2f}".format(x)


def str4(x):
    return "{:.4f}".format(x)


def dirfiles(img_path,img_ext):
    img_names = fnmatch.filter(sorted(listdir(img_path)),img_ext)
    return img_names


def read_img(img_path):
  '''
  if model_name=='dlib':
    img    = cv2.imread(img_path) ## reading image
  elif model_name=='lbp':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256)) 
  elif model_name=='arcface':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256,256)) 

  if model_name=='vggface2':
    img = cv2.imread(img_path)
    # convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256)) 
  elif model_name=='casia_webface':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256)) 

  elif model_name=='arcface':
    img = cv2.imread(img_path)
    img = cv2.resize(img, (256, 256))
  '''

  img = cv2.imread(img_path)
  # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
  img = cv2.resize(img, (256, 256))

  return img



def put_label(X,label,w,scale=0.75,thk=1,color=(0,0,0),rotate=0):
    (i1,i2,j1,j2,d) = w
    if rotate==0:
        T = 255*np.ones((i2-i1,j2-j1,3),np.uint8)
        cv2.putText(T,text= label, org=(d,i2-i1-d),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=color,
            thickness=thk, lineType=cv2.LINE_AA)
    else:
        T = 255*np.ones((j2-j1,i2-i1,3),np.uint8)
        cv2.putText(T, text= label, org=(d,j2-j1-d),
            fontFace= cv2.FONT_HERSHEY_SIMPLEX, fontScale=scale, color=color,
            thickness=thk, lineType=cv2.LINE_AA)
        if rotate>0:
            T = cv2.rotate(T, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            T = cv2.rotate(T, cv2.cv2.ROTATE_90_CLOCKWISE)
    X[i1:i2,j1:j2] = (X[i1:i2,j1:j2]+1.5*T)/2.5


def imshow(I,height=6,width=None,show_pause=0,title=None,fpath=None):
  n = height
  if width!=None:
    m = width
  else:
    N = I.shape[0]
    M = I.shape[1]
    m = round(n*M/N)
  __,ax = plt.subplots(1,1,figsize=(m,n))
  bw = len(I.shape)==2
  if bw:
     ax.imshow(I)
  else:
     ax.imshow(cv2.cvtColor(I, cv2.COLOR_BGR2RGB))

  if title!=None:
    ax.set_title(title)
  plt.axis('off')
  if fpath!=None:
     plt.savefig (fpath,bbox_inches = 'tight',pad_inches = 0)
     

  if show_pause>0:
    plt.pause(show_pause)
    plt.close()
  else:
    plt.show()


def heatmap(Ao,H,ss,alpha=0.5,type='Max'):
    hm = gaussian_kernel(ss,ss/8.5,type='Max')
    X  = cv2.filter2D(H,-1,hm)
    D = X-np.min(X)
    if type=='Max':
        #D = minmax_norm(X)
        D = D/np.max(D)
    else:
        D = D/np.sum(D)
    X  = np.uint8(D*255)
    HM = cv2.applyColorMap(X, cv2.COLORMAP_JET)
    Y  = cv2.addWeighted(HM, alpha, Ao, 1-alpha, 0)
    return D,Y,HM


def contours(D,A,color_map,contour_levels, print_levels=True,color_levels=True,contour_save_path=None,save_with_face=True,):
    # D: heatmap
    # A: background image
    # Examples
    # contours(D,A,'jet'  ,10,print_levels=False,color_levels=True,img_file='Cont.png')
    # contours(D,A,'white',10,print_levels=True,color_levels=False,img_file=None)
    height    = D.shape[0]
    width     = D.shape[1]
    levels    = np.linspace(0.1, 1.0, contour_levels)
    x         = np.arange(0, width, 1)
    y         = np.arange(0, height, 1)
    extent    = (x.min(), x.max(), y.min(), y.max())

    Z = D/D.max()

    # creating blank image to get just the contours
    if not save_with_face:
      A=np.zeros((height,width,3),np.uint8)


    At = cv2.cvtColor(A, cv2.COLOR_BGR2RGB)
    plt.figure()
    plt.imshow(At,extent=extent)

    if color_levels:
      CS = plt.contour(Z, levels, cmap=color_map, origin='upper', extent=extent)
    else:
      CS = plt.contour(Z, levels, colors=color_map, origin='upper', extent=extent)
    


    if print_levels:
      plt.clabel(CS,fontsize=9, inline=1)
    plt.axis('off')
    plt.savefig (contour_save_path,bbox_inches = 'tight',pad_inches = 0)
    # plt.show()
    plt.clf()
    plt.close()
    C = cv2.imread(contour_save_path)
    C = cv2.resize(C,(width,height))
    return C


## Image Processing
def gaussian_kernel(size, sigma, type='Sum'):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1,
           -size // 2 + 1:size // 2 + 1]
    kernel = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    if type=='Sum':
      kernel = kernel / kernel.sum()
    else:
      kernel = kernel / kernel.max()
    return kernel.astype('double')


def minmax(X, low, high, minX=None, maxX=None, dtype=float):
    X = np.asarray(X)
    if minX is None:
        minX = np.min(X)
    if maxX is None:
        maxX = np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high - low)
    X = X + low
    return np.asarray(X, dtype=dtype)


def minmax_norm(X):
  Y = X-X.min()
  Y = Y/Y.max()
  return Y


def DefineGaussianMask(ic,jc,nh,N=256,M=256):
  # Define an image of NxM, with a Gaussian of nh x nh centred in (ic,jc)
  nh2 = round(nh/2)
  i1  = ic
  j1  = jc
  n   = N+nh
  m   = M+nh
  s   = nh/8.5
  h   = 1-gaussian_kernel(nh,s,type='Max')
  Mk  = np.ones((n,m))
  i2  = i1+nh
  j2  = j1+nh
  Mk[i1:i2,j1:j2] = h
  return Mk[nh2:nh2+N,nh2:nh2+M]


def MaskMult(A,Mk):
    n = Mk.shape[0]
    m = Mk.shape[1]
    M = np.zeros((n,m,3))
    M[:,:,0] = Mk
    M[:,:,1] = Mk
    M[:,:,2] = Mk
    Ak = np.multiply(M,A)
    return Ak.astype(np.uint8)


def MaskMultNeg(BB,Bo,Mk):
    B1 = MaskMult(Bo,1-Mk)
    Bn = (255-B1)/255
    Bk = 255 - np.multiply(BB,Bn)
    return Bk.astype(np.uint8)


## Face Recognition
def extract_hist(img,norm=True):
    hists = []
    num_points = 8
    radii = [1, 2]
    grid_x = 9
    grid_y = 9

    for radius in radii:
        lbp = local_binary_pattern(img,
                                   num_points,
                                   radius, 'nri_uniform')

        height = lbp.shape[0] // grid_x
        width = lbp.shape[1] // grid_y
        indices = itertools.product(range(int(grid_x)),
                                    range(int(grid_y)))
        for (i, j) in indices:
            top = i * height
            left = j * width
            bottom = top + height
            right = left + width
            region = lbp[top:bottom, left:right]
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(region, density=True,
                                   bins=n_bins,
                                   range=(0, n_bins))
            hists.append(hist)

    hists = np.asarray(hists)

    x = np.ravel(hists)
    if norm:
      x = x/np.linalg.norm(x)

    return x


class TanTriggsProc():
    def __init__(self, alpha=0.1, tau=10.0, gamma=0.2, sigma0=2.0, sigma1=3.0):
        self._alpha = float(alpha)
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._sigma0 = float(sigma0)
        self._sigma1 = float(sigma1)

    def compute(self, X):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        X = np.array(X, dtype=np.float32)
        X = np.power(X, self._gamma)
        X = np.asarray(ndimage.gaussian_filter(X, self._sigma1) - ndimage.gaussian_filter(X, self._sigma0))
        X = X / np.power(np.mean(np.power(np.abs(X), self._alpha)), 1.0 / self._alpha)
        X = X / np.power(np.mean(np.power(np.minimum(np.abs(X), self._tau), self._alpha)), 1.0 / self._alpha)
        X = self._tau * np.tanh(X / self._tau)
        return X

tantriggs = TanTriggsProc()



def face_embedding(img, binary_comparison=False):
    # if model_name=='lbp':
    #     imgr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #     imgr = cv2.resize(imgr,(108,108)) ##return the reshaped image
    #     x   = extract_hist(imgr)
    # elif model_name == 'dlib':
    #     fl = [[0,len(img)-1,len(img[0])-1,0]]
    #     x = face_model.face_encodings(img,fl)[0]
    #     x = x/np.linalg.norm(x)

    ################## FACENET ##################
    # if model_name == 'vggface2':
    #     timg = torch.tensor(img)
    #     timg = timg.permute(2, 0, 1)
    #     timg = (timg - 127.5) / 128.0
    #     z = timg.unsqueeze(0)
    #     a = face_model(z)
    #     x = a.detach().numpy()
    #     x = x.reshape((512,))
    
    # elif model_name == 'casia_webface':
    #     timg = torch.tensor(img)
    #     timg = timg.permute(2, 0, 1)
    #     timg = (timg - 127.5) / 128.0
    #     z = timg.unsqueeze(0)
    #     a = face_model(z)
    #     x = a.detach().numpy()
    #     x = x.reshape((512,))
    
    ################## DEEPFACE ##################

    # error in ['DeepFace', 'Dlib', "Facenet", "Facenet512",'SFace', "DeepID",]
    # if model_name in ["VGG-Face", "OpenFace", "ArcFace"] and detector_name in ["mtcnn"]:    

    embedding = DeepFace.represent(
                    img, 
                    model_name=model_name, 
                    detector_backend=detector_name, 
                    enforce_detection=False, 
                    align=True,
                    
        )[0]["embedding"]

    return embedding


## Adding/Removing Masks
def removing_score(A,B,d,nh,ROI=None,background=1.0):
  N      = A.shape[0]
  M      = A.shape[1]
  xA     = face_embedding(A)
  Hsc    = background*np.ones((N,M))
  sc_min = 10
  sc_max = -10
  if ROI is None:
      ROI = np.ones((N,M,3))

  #if len(ROI.shape)==2:
  #    ROI = np.dstack((ROI,ROI,ROI))
  for ic in range(d,N,d):
    for jc in range(d,M,d):
      if np.sum(ROI[ic,jc,:])>0:
        Mk  = DefineGaussianMask(ic,jc,nh)
        Bk  = MaskMult(B,Mk)
        xBk = face_embedding(Bk)
        sc  = np.dot(xA,xBk)
        Hsc[ic,jc] = sc


        if sc<sc_min:
            sc_min  = sc
            out_min = [Bk,sc_min,ic,jc]
        if sc>sc_max:
            sc_max  = sc
            out_max = [Bk,sc_max,ic,jc]
  return Hsc,out_min,out_max


def adding_score(A,B,Bo,d,nh,ROI=None):
  N      = A.shape[0]
  M      = A.shape[1]
  #xA     = arcface.calc_emb(A)
  #xB     = arcface.calc_emb(B)
  xA     = face_embedding(A)
  xB     = face_embedding(B)
  nh2    = round(nh/2)
  sc_max = -10
  sc_min = 10
  BB     = 255-B
  Hsc    = np.zeros((N,M))
  if ROI is None:
      ROI = np.ones((N,M,3))
  for ic in range(d,N,d):
    for jc in range(d,M,d):
      if np.sum(ROI[ic,jc,:])>0:
        Mk = DefineGaussianMask(ic,jc,nh)
        Bk = MaskMultNeg(BB,Bo,Mk)
        #xB = arcface.calc_emb(Bk)
        xB = face_embedding(Bk)
        sc = np.dot(xA,xB)
        Hsc[ic,jc] = sc
        if sc<sc_min:
           sc_min  = sc
           out_min = [Bk,sc_min,ic,jc]
        if sc>sc_max:
           sc_max  = sc
           out_max = [Bk,sc_max,ic,jc]
  return Hsc,out_min,out_max


def remove_best(A,B,d,nh):
  Hsc,__,out_max = removing_score(A,B,d,nh,ROI=B)
  Bks = out_max[0]
  output = out_max[1:]
  return Bks,Hsc,output


def remove_worst(A,B,d,nh):
  Hsc,out_min,__ = removing_score(A,B,d,nh)
  Bks = out_min[0]
  output = out_min[1:]
  return Bks,Hsc,output


def add_best(A,B,Bo,d,nh):
  Hsc,__,out_max = adding_score(A,B,Bo,d,nh)
  Bks = out_max[0]
  output = out_max[1:]
  return Bks,Hsc,output


def saliency_minus(A,B,nh,d,n,nmod,th, display=True):
    N    = A.shape[0]
    M    = A.shape[1]  
    H1   = np.zeros((N,M))
    xA   = face_embedding(A)
    xB   = face_embedding(B)
    sc0  = np.dot(xA,xB)
    st   = "{:7.4f}".format(sc0)
    # print('minus t = 000 sc[0]='+st)
    
    if display: 
      imshow(cv2.hconcat([A,B]),show_pause=1,title='sc0 = '+st)
    # else: print('Similarity: ', st)
    
    t    = 0
    sct  = sc0
    Bt   = B
    dsc  = 1
    t0   = time.time()
    while dsc>th and t<n: # and (t==0 or sct>0): 
        t = t+1
        Hsc,out_min,__ = removing_score(A,Bt,d,nh,ROI=Bt,background=sc0)
        if t==1:
            H0 = sc0-Hsc
        sct     = out_min[1]       # minimal score by removing a gaussian mask centered in (i,j)
        i       = out_min[2]
        j       = out_min[3]
        sct_st  = "{:7.4f}".format(sct)
        dsc     = sc0-sct
        H1[i,j] = dsc
        sc0 = sct
        hij_st = "{:.4f}".format(H1[i,j])
        t1 = time.time()  
        dt_st = "{:.2f}".format(t1-t0)
        t0     = t1  
        # print('minus t = '+num2fixstr(t,3)+' sc[t]='+sct_st+ ' H[' + num2fixstr(i,3)+ ',' +num2fixstr(j,3) + '] = sc[t-1]-sc[t] = '+hij_st+' > '+dt_st+'s')
        Bt      = out_min[0]
        if np.mod(t,nmod)==0:
            imshow(Bt,show_pause=1)
    return H0,H1,st


def saliency_plus(A,B,nh,d,n,nmod,th, display=True):
    N    = A.shape[0]
    M    = A.shape[1]  
    Bt   = np.random.rand(N,M,3)*2
    Bt   = Bt.astype(np.uint8)
    H1   = np.zeros((N,M))
    xA   = face_embedding(A)
    xB   = face_embedding(Bt)
    sc0  = np.dot(xA,xB)
    st   = "{:7.4f}".format(sc0)
    # print('plus  t = 000 sc[0]='+st)
    if display: imshow(cv2.hconcat([A,B]),show_pause=1)
    sct  = sc0
    t    = 0
    dsc  = 1
    t0   = time.time()
    while dsc>th and t<n: # and (t==0 or sct>0):
        t = t+1
        Bt,Hsc,out_max = add_best(A,Bt,B,d,nh)
        sct = out_max[0]
        if t==1:
            H0 = Hsc-sc0
        i       = out_max[1]
        j       = out_max[2]
        sct_st  = "{:7.4f}".format(sct)
        dsc     = sct-sc0
        H1[i,j] = dsc
        sc0     = sct
        hij_st  = "{:.4f}".format(H1[i,j])
        t1 = time.time()  
        dt_st = "{:.2f}".format(t1-t0)
        t0     = t1  
        # print('plus  t = '+num2fixstr(t,3)+' sc[t]='+sct_st+ ' H[' + num2fixstr(i,3)+ ',' +num2fixstr(j,3) + '] = sc[t]-sc[t-1] = '+hij_st+' > '+dt_st+'s')
        if np.mod(t,nmod)==0:
            imshow(Bt,show_pause=1)
    return H0,H1,st


def DefineRISEmasks(k,N,M,smin,smax,kernel='Gauss'):
  height = 2*N
  width  = 2*M
  ii = randomints(0, height, k)
  jj = randomints(0, width, k)
  ss = randomints(smin,smax,k)
  Mk  = np.ones((height,width))
  for t in range(len(ii)):
      i1 = ii[t]
      j1 = jj[t]
      s = 1.0*ss[t]
      if kernel=='Gauss':
         h  = 1-gaussian_kernel(s,s/8.5,type='Max')
      else:
         h = 1.0*np.zeros((ss[t],ss[t]))
      n = h.shape[0]
      i2 = i1+n
      j2 = j1+n
      if i2>height:
        i2 = height
      if j2>width:
        j2 = width
      Mk[i1:i2,j1:j2] = np.multiply(Mk[i1:i2,j1:j2],h[0:i2-i1,0:j2-j1])

  i1 = round(N/2)
  j1 = round(M/2)
  if kernel=='Square':
    Mks = cv2.resize(Mk,(32,32))
    Mk  = cv2.resize(Mks,(height,width))
  return Mk[i1:i1+N,j1:j1+M]


def saliency_RISE(A,B,n,k,smin,smax,kernel='Gauss'):
    N      = A.shape[0]
    M      = A.shape[1]
    xA     = face_embedding(A)
    S      = np.zeros((N,M))
    sc_max = 10
    for t in tqdm(range(n)):
      Mk  = 1-DefineRISEmasks(k,N,M,smin,smax,kernel=kernel)
      Bt  = MaskMult(B,Mk)
      xBt = face_embedding(Bt)
      sc  = np.dot(xA,xBt)
      S   = S+sc*Mk
      if sc<sc_max:
        sc_max = sc
    print('sc_min=',sc)
    S = S/n
    howis(S)
    return S


def perturbations(n,m,p=0.5):
    perts = np.random.binomial(1, p, size=(n,m))
    return perts


def perturb_image(img,perturbation,segments):
    active_pixels = np.where(perturbation == 1)[0]
    mask = np.zeros(segments.shape)
    for active in active_pixels:
        mask[segments == active] = 1 
    perturbed_image = copy.deepcopy(img)
    X = perturbed_image*mask[:,:,np.newaxis]
    X = X.astype(np.uint8)
    return X,mask


def color_mask(X,M,col):
    Y = copy.deepcopy(X)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i,j]==0:
                Y[i,j,:]=col
    return Y


def show_superpixels(B,superpixels):
    Bs = skimage.segmentation.mark_boundaries(B, superpixels)
    Bs = Bs*255
    Bs = Bs.astype(np.uint8)
    return Bs


def LIME_saliency(A,B,N=500,num_top_features=16,kernel_size=3,max_dist=200, ratio=0.2):
    xA          = face_embedding(A)
    xB          = face_embedding(B)
    sc          = np.dot(xA,xB)
    print('score = '+str4(sc))
    superpixels = skimage.segmentation.quickshift(B, kernel_size=kernel_size,max_dist=max_dist, ratio=ratio)
    m           = np.unique(superpixels).shape[0]
    perts       = perturbations(N,m)
    Y = cv2.hconcat([A,B])
    imshow(Y)
    scores = []
    s = np.zeros((B.shape[0],B.shape[1]))
    for i in tqdm(range(N)):
        Bt,Mt = perturb_image(B,perts[i],superpixels)
        xB    = face_embedding(Bt)
        sc    = np.dot(xA,xB)
        s     = s + sc*Mt
        scores.append(sc)
    s = s/N
    predictions = np.array(scores)
    original_image = np.ones(m)[np.newaxis,:] #Perturbation with all superpixels enabled 
    distances      = sklearn.metrics.pairwise_distances(perts,original_image, metric='cosine').ravel()
    kernel_width   = 0.25
    weights        = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
    simpler_model  = LinearRegression()
    simpler_model.fit(X=perts, y=predictions, sample_weight=weights)
    coeff          = simpler_model.coef_
    ys             = simpler_model.predict(perts)
    err            = np.abs(ys-predictions)
    print('error = '+str4(np.mean(err))) 
    top_features   = np.argsort(coeff)[-num_top_features:] 
    mask = np.zeros(m) 
    mask[top_features]= True #Activate top superpixels
    Bs,Ms = perturb_image(B,mask,superpixels)
    return Bs,Ms,s,superpixels


def SEQ_saliency(A,B,nh,d,n,nmod,add_mask,K,th1,th0):
  # Switch between adding/removing mask keeping the score between th0 and th1
  # In each switch nh=nk*K and d=d*K 
  # nh       = 200  # gaussian mask
  # d        = 32   # stride
  # n        = 20   # number of iterations
  # nmod     = 10   # display results each nmod iterations
  # add_mask = True # add/remove
  # K        = 0.9  # reduce factor
  # th1      = 0.85 # threshold for adding
  # th0      = 0.80 # threshold for removing


    lst = ['removing','adding  ']
    xA  = face_embedding(A)
    xB  = face_embedding(B)
    sc0 = np.dot(xA,xB)

    N   = A.shape[0]
    M   = A.shape[1]  

    if add_mask:
        Bt = np.zeros((N,M,3))
        Bt = Bt.astype(np.uint8)
    else:
        Bt = B

    sc = 0
    qq = 0
    nq = 15
    t = 0
    ps = 0
    while t<=n and ps<5:
        t = t+1
    #for t in range(n):
        sc_str = "{:.6f}".format(sc)
        nh_str = "({:3d},{:3d})".format(nh,d)
        print(num2fixstr(t,4)+'/'+num2fixstr(n,4)+': '+lst[1*add_mask]+' with '+nh_str+': score='+sc_str+' ths='+str2(th0)+','+str2(th1))
        add_mask_0 = add_mask
        if add_mask:
            Bt,H,out = add_best(A,Bt,B,d,nh)
            sc = out[0]
            qq = qq+1
            add_mask = sc < th1
        else: 
            Bt,H,out = remove_best(A,Bt,d,nh)
            sc = out[0]
            qq = qq+1
            add_mask = sc < th0 

        if (add_mask != add_mask_0) or (qq==nq):
            nh = round(nh*K)
            d  = round(d*K)
            qq = 0
            nq = nq-2
            ps = ps+1
            add_mask = 1-add_mask_0
            print('switch '+str(ps))
        if np.mod(t,nmod)==0:
            Y  = cv2.hconcat([A,Bt])
            st = 'score('+str(t)+') ='+str(sc)
            imshow(Y,show_pause=0.1,title=st)

    xA = face_embedding(A)
    xB = face_embedding(Bt)
    sc = np.dot(xA,xB)
    Y  = cv2.hconcat([A,Bt])
    st = 'final score = '+str(sc)
    print(st)
    return H,Bt
