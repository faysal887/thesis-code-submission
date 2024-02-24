
import os, cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import utils.blackbox_explaination.fvx_utils as fxv_utils

pos       = (15,25)
dsc       = 0.05

FAST_MODE = True

if FAST_MODE:  # 1 minute
    gsigma    = 221                # width of Gaussian mask
    d         = 48                 # steps (one evaluation each d x d pixeles)
    tmax      = 1                  # maximal number of iterations
else:          # 6 minutes
    gsigma    = 161                # width of Gaussian mask
    d         = 8                  # steps (one evaluation each d x d pixeles)
    tmax      = 20                 # maximal number of iterations



def get_input_images(img_A, img_B, display=True):
    A         = fxv_utils.read_img(img_A)
    Bo        = fxv_utils.read_img(img_B)
    (N,M)     = A.shape[0:2]

    As        = A.copy()
    Bs        = Bo.copy()
    cv2.putText(As,'A', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(Bs,'B', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    Y         = cv2.hconcat([As,Bs])
    if display: fxv_utils.imshow(Y,title='input images')

    return A, Bo, As, Bs


def get_sliency_minus_single(A,Bo,As,Bs, save_path,contour_face_save_path,contour_no_face_save_path,display=True):
    H0m,H1m,simialrity   = fxv_utils.saliency_minus(A,Bo,nh=gsigma,d=d,n=tmax,nmod=2,th=dsc, display=False)

    S0m,Y0m, HM_S0_minus   = fxv_utils.heatmap(Bo,H0m,gsigma)
    cv2.putText(Y0m,'S0-', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    # Y         = cv2.hconcat([As,Bs,Y0m,Y1m])

    # with face - single
    C_face         = fxv_utils.contours(
                    255*S0m,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - single
    C_no_face         = fxv_utils.contours(
                    255*S0m,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )

    Y         = cv2.hconcat([As,Bs,Y0m,C_face])

    if display: fxv_utils.imshow(Y,fpath=save_path,show_pause=1)
    return S0m, HM_S0_minus


def get_sliency_minus_greedy(A,Bo,As,Bs, save_path,contour_face_save_path,contour_no_face_save_path,display=True):
    H0m,H1m,simialrity   = fxv_utils.saliency_minus(A,Bo,nh=gsigma,d=d,n=tmax,nmod=2,th=dsc, display=False)
    S1m,Y1m, HM_S1_minus   = fxv_utils.heatmap(Bo,H1m,gsigma)
    cv2.putText(Y1m,'S1-', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    
    # with face - greedy
    C_face         = fxv_utils.contours(
                    255*S1m,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - greedy
    C_no_face       = fxv_utils.contours(
                    255*S1m,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )
    
    Y         = cv2.hconcat([As,Bs,Y1m,C_face])

    
    if display: fxv_utils.imshow(Y,fpath=save_path,show_pause=1)
    return S1m, HM_S1_minus


def get_sliency_plus_single(A,Bo,As,Bs, save_path,contour_face_save_path,contour_no_face_save_path,display=True):
    H0p,H1p,simialrity   = fxv_utils.saliency_plus(A,Bo,nh=gsigma,d=d,n=tmax,nmod=2,th=dsc, display=False)
    S0p,Y0p, HM_S0_plus   = fxv_utils.heatmap(Bo,H0p,gsigma)
    cv2.putText(Y0p,'S0+', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # with face - single
    C_face         = fxv_utils.contours(
                    255*S0p,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )

    # without face - single
    C_no_face         = fxv_utils.contours(
                    255*S0p,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )
    
    Y         = cv2.hconcat([As,Bs,Y0p,C_face])

    
    if display: fxv_utils.imshow(Y,fpath=save_path,show_pause=1)
    return S0p,HM_S0_plus


def get_sliency_plus_greedy(A,Bo,As,Bs, save_path,contour_face_save_path,contour_no_face_save_path,display=True):
    H0p,H1p,simialrity   = fxv_utils.saliency_plus(A,Bo,nh=gsigma,d=d,n=tmax,nmod=2,th=dsc, display=False)
    S1p,Y1p, HM_S1_plus   = fxv_utils.heatmap(Bo,H1p,gsigma)
    cv2.putText(Y1p,'S1+', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # with face - greedy
    C_face         = fxv_utils.contours(
                    255*S1p,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - greedy
    C_no_face         = fxv_utils.contours(
                    255*S1p,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )
    

    Y         = cv2.hconcat([As,Bs,Y1p,C_face])


    if display: fxv_utils.imshow(Y,fpath=save_path,show_pause=1)
    return S1p,HM_S1_plus


def get_sliency_avg(S0m,S0p,S1m,S1p, A,Bo,As,Bs, merged_save_path, face_heatmap_save_path, contour_face_save_path,contour_no_face_save_path,display=True):
    Savg      = (S0m+S0p+S1m+S1p)/4 # <= HeatMap between 0 and 1
    _,Yavg,HM    = fxv_utils.heatmap(Bo,Savg,199)
    # cv2.imwrite('/data/faysal/thesis/tmp/Savg.png',Savg)
    # plt.imshow(D)
    # plt.imshow(X)

    # with face
    C_face         = fxv_utils.contours(
                    255*Savg,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=contour_face_save_path, 
                    save_with_face=True
                )
    
    # without face
    C_no_face         = fxv_utils.contours(
                    255*Savg,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=contour_no_face_save_path, 
                    save_with_face=False
                )
    
    cv2.putText(Yavg,'AVG' , pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(C_face ,'AVGc', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    Y         = cv2.hconcat([As,Bs,Yavg,C_face])
    # print(face_heatmap_save_path)

    if not os.path.exists(os.path.dirname(face_heatmap_save_path)): os.mkdir(os.path.dirname(face_heatmap_save_path))
    cv2.imwrite(face_heatmap_save_path,Yavg)
    # cv2.imwrite(face_heatmap_save_path,HM)


    if display: fxv_utils.imshow(Y,fpath=merged_save_path,show_pause=1)
    if merged_save_path: cv2.imwrite(merged_save_path,Y)

    return HM


def get_saliency_seq(A,Bo,As,Bs, merged_save_path, contour_face_save_path,contour_no_face_save_path, display=True):
    nh       = 400    # gaussian mask
    d        = 32     # stride
    n        = 100    # number of iterations
    nmod     = 20      # display results each nmod iterations
    add_mask = False  # true:add / false:remove
    K        = 0.9    # reduce factor
    B  = Bo.copy()
    xA = fxv_utils.face_embedding(A)
    xB = fxv_utils.face_embedding(B)
    sc = np.dot(xA,xB)
    th1  = sc+0.2 # threshold for adding
    th0  = sc-0.2 # threshold for removing

    if th1>1:
        th1 = 0.9
    if th0<0:
        th0 = 0.1

    print('score = '+str4(sc))
    fxv_utils.imshow(cv2.hconcat([A,B]))
    (H,Bt) = fxv_utils.SEQ_saliency(A,B,nh,d,n,nmod,add_mask,K,th1,th0)

    D,Yseq,HM = fxv_utils.heatmap(B,sc-H,128)
    Bss = copy.deepcopy(B)

    N = Bt.shape[0]
    M = Bt.shape[1]
    for i in range(N):
        for j in range(M):
            if np.sum(Bt[i,j,:])<50:
                Bss[i,j,:] = [64,64,64]

    cv2.putText(Yseq,'SEQ' , pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(Bss ,'Region' , pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)


    # with face - greedy
    C_face         = fxv_utils.contours(
                    255*Yseq,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - greedy
    C_no_face         = fxv_utils.contours(
                    255*Yseq,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )


    Y = cv2.hconcat([As,Bs,Bss,Yseq,C_face])

    # imshow(Y,fpath='heatmap_SEQ.png',show_pause=1)
    if display: fxv_utils.imshow(Y,fpath=merged_save_path,show_pause=1)
    if merged_save_path: cv2.imwrite(merged_save_path,Y)

    return HM


def get_saliency_rise_gauss(A,Bo,As,Bs,merged_save_path,contour_face_save_path,contour_no_face_save_path,display=True): 
    S = fxv_utils.saliency_RISE(A,Bo,100,40,60,250,kernel='Gauss')
    _,Y, HM = fxv_utils.heatmap(Bo,S,99)
    cv2.putText(Y,'RISEgauss', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # with face - greedy
    C_face         = fxv_utils.contours(
                    255*S,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - greedy
    C_no_face         = fxv_utils.contours(
                    255*S,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )

    # Y = cv2.hconcat([As,Bs,Ygauss,Ysquare])
    Y = cv2.hconcat([As,Bs,Y,C_face])
    if display: fxv_utils.imshow(Y,fpath=merged_save_path,show_pause=1)
    if merged_save_path: cv2.imwrite(merged_save_path,Y)

    return HM


def get_saliency_rise_square(A,Bo,As,Bs,merged_save_path,contour_face_save_path,contour_no_face_save_path,display=True): 
    S = fxv_utils.saliency_RISE(A,Bo,100,40,20,80,kernel='Square')
    _,Y, HM = fxv_utils.heatmap(Bo,S,99)
    cv2.putText(Y,'RISEsquare', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)

    # with face - greedy
    C_face         = fxv_utils.contours(
                    255*S,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - greedy
    C_no_face         = fxv_utils.contours(
                    255*S,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )

    # Y = cv2.hconcat([As,Bs,Ygauss,Ysquare])
    Y = cv2.hconcat([As,Bs,Y,C_face])
    if display: fxv_utils.imshow(Y,fpath=merged_save_path,show_pause=1)
    if merged_save_path: cv2.imwrite(merged_save_path,Y)

    return HM
    g


def get_saliency_lime(A,Bo,As,Bs, merged_save_path,contour_face_save_path,contour_no_face_save_path, N=500 ,display=True):
    Bsp,Ms,X,superpixels = fxv_utils.LIME_saliency(A,Bo,N)
    D1,Ylime,HM = fxv_utils.heatmap(Bo,X,97)
    Bk = fxv_utils.color_mask(Bo,Ms,[64,64,64])
    Bss = fxv_utils.show_superpixels(Bo,superpixels)
    cv2.putText(Bss,'Superpixels', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(Bk,'Mask', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)
    cv2.putText(Ylime,'LIME', pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255),2)


    # with face - greedy
    C_face         = fxv_utils.contours(
                    255*X,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_face_save_path}', 
                    save_with_face=True
                )
    
    # without face - greedy
    C_no_face         = fxv_utils.contours(
                    255*X,Bo,'jet',10, 
                    print_levels=False,
                    color_levels=True, 
                    contour_save_path=f'{contour_no_face_save_path}', 
                    save_with_face=False
                )

    Y = cv2.hconcat([As,Bs,Bss,Bk,Ylime,C_face])
    
    # imshow(Y,fpath='heatmap_LIME.png')
    
    if display: fxv_utils.imshow(Y,fpath=merged_save_path,show_pause=1)
    if merged_save_path: cv2.imwrite(merged_save_path,Y)
    
    return HM


def get_img_vector(fn):
    img = cv2.imread(fn)

    timg = torch.tensor(img)
    timg = timg.permute(2, 0, 1)
    timg = (timg - 127.5) / 128.0
    z = timg.unsqueeze(0)
    # a = face_model(z)
    # x = a.detach().numpy()
    # x = x.reshape((512,))
    # return x
    return z


def get_images_avg(fn1, fn2, save_path, save=False, display=True):
    # Load the two input images
    image1_o = cv2.imread(fn1)
    image2_o = cv2.imread(fn2)

    # Ensure that both images have the same dimensions
    if image1_o.shape != image2_o.shape:
        raise ValueError("Both input images must have the same dimensions")

    # Convert the images to float32 for accurate computation
    image1 = image1_o.astype(np.float32)
    image2 = image2_o.astype(np.float32)

    # Perform the Average Operation
    combined_saliency = 0.5 * (image1 + image2)

    # Convert the result back to uint8 format (0-255)
    combined_saliency = np.round(combined_saliency).astype(np.uint8)

    image1_o = cv2.cvtColor(image1_o, cv2.COLOR_BGR2RGB)
    image2_o = cv2.cvtColor(image2_o, cv2.COLOR_BGR2RGB)
    combined_saliency = cv2.cvtColor(combined_saliency, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('output/average_saliency.jpg', combined_saliency)

    merged_saliency = cv2.hconcat([image1_o,image2_o,combined_saliency])
    if display: 
        plt.imshow(merged_saliency)
    if save: 
        cv2.imwrite(save_path,merged_saliency)
   

    return combined_saliency


def recreate_folder(path):
    try: shutil.rmtree(path)
    except: pass
    os.mkdir(path)
    os.mkdir(f"{path}/face_cropped")
    os.mkdir(f"{path}/face_heatmap")
    os.mkdir(f"{path}/other")

