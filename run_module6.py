"""Actual public-data optical flow and four-view fiducial-assisted reconstruction.

The source photographs and videos are NOT Aya's captures. World distances are
in chessboard-square units, never invented metres. Boundary corners are withheld
from camera pose fitting. See README and the generated report for limitations.
"""
from pathlib import Path
import cv2, numpy as np, json, subprocess, sys, shutil, csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio_ffmpeg

ROOT=Path(__file__).resolve().parent;D=ROOT/'data';O=ROOT/'results';O.mkdir(exist_ok=True)
sys.path.insert(0,str(ROOT));import cv_course
def js(p,x):p.write_text(json.dumps(x,indent=2))
def prepare_clips():
    """Keep native timestamps/frame rates; scale each frame to width 640."""
    for original,name,start in [('vtest.avi','pedestrians',10),('v_JspVuT6rsLA.mp4','violin',45)]:
        dest=D/(name+'_30s.mp4')
        if not dest.exists():
            subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-ss',str(start),'-i',str(D/original),'-t','30','-vf','scale=640:-2','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p',str(dest)],check=True)
    # Independent visual annotations recorded before running either tracker.
    # Coordinates refer to the scaled original consecutive frames at t=15/50s.
    # Computer-assisted visual inspection; integer precision, approximately +/-3 px.
    annotations={
      'pedestrians':{'points_first':[[503,194],[342,168],[377,167],[345,169],[358,200],[422,284]],
                     'points_next':[[498,194],[343,168],[378,168],[345,169],[358,200],[422,284]],
                     'descriptions':['moving pedestrian head','central left pedestrian head','central right pedestrian head','sign top-left corner (static)','sign bottom-right corner (static)','tripod top (static)']},
      'violin':{'points_first':[[312,54],[417,184],[95,239],[116,260],[275,119],[470,40]],
                 'points_next':[[314,69],[417,186],[94,245],[116,260],[275,119],[470,40]],
                 'descriptions':['bow upper tip (moving, blur)','violin scroll (moving)','bow-arm elbow (moving)','keyboard edge (static)','piano corner (static)','lamp corner (static)']}}
    js(O/'manual_annotations.json',{'provenance':'computer-assisted visual inspection, not tracker-generated ground truth','estimated_coordinate_uncertainty_px':3,'annotations':annotations})
    cfg={'out':str(O/'flow'),'videos':[]}
    for name,fps in [('pedestrians',10),('violin',25)]:
        a=annotations[name];cfg['videos'].append({'path':str(D/(name+'_30s.mp4')),'seconds':30,'start_seconds':0,'validation_pair_index':5*fps,'display_max_px_per_frame':8,'manual_points_first':a['points_first'],'manual_points_next':a['points_next']})
    js(ROOT/'flow_config.json',cfg);cv_course.flow(cfg)
    for i in (1,2):
        f=O/'flow'/f'video_{i}'/'optical_flow.mp4';tmp=f.with_name('optical_flow_h264.mp4')
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(f),'-c:v','libx264','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(tmp)],check=True)
        tmp.replace(f)

def reconstruct():
    obj=np.zeros((54,3),np.float32);obj[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    corners={}; rejected=[]
    for f in sorted(D.glob('left*.jpg')):
        ok,c=cv2.findChessboardCornersSB(cv2.imread(str(f),0),(9,6))
        if ok: corners[f.name]=c.reshape(-1,2)
        else: rejected.append(f.name)
    views=['left01.jpg','left02.jpg','left06.jpg','left07.jpg']
    train=[n for n in corners if n not in views]
    rms,K,dist,_,_=cv2.calibrateCamera([obj]*len(train),[corners[n][:,None,:] for n in train],(640,480),None,None)
    np.savez(O/'calibration.npz',K=K,dist=dist,image_size=np.array([640,480]))
    # Pose uses only the outer ring of detected inner corners; target polygon
    # corners lie strictly inside and are withheld from every pose fit.
    pose_ids=np.where((obj[:,0]==0)|(obj[:,0]==8)|(obj[:,1]==0)|(obj[:,1]==5))[0]
    xy=np.array([[2,1],[4,1],[6,1],[6,2],[6,4],[4,4],[2,4],[2,2]])
    ids=np.array([np.where(np.all(obj[:,:2]==p,axis=1))[0][0] for p in xy])
    assert not set(ids)&set(pose_ids)
    Ps=[];norm=[];poses=[];pix=[]
    for name in views:
        pts=corners[name].astype(float)
        ok,r,t=cv2.solvePnP(obj[pose_ids].astype(float),pts[pose_ids],K,dist);assert ok
        R=cv2.Rodrigues(r)[0];P=np.column_stack([R,t]);Ps.append(P)
        norm.append(cv2.undistortPoints(pts[ids,None,:],K,dist).reshape(-1,2));pix.append(pts[ids])
        projected=cv2.projectPoints(obj[pose_ids],r,t,K,dist)[0].reshape(-1,2)
        poses.append({'file':name,'R':R.tolist(),'t_board_squares':t.ravel().tolist(),'camera_center_board_squares':(-R.T@t).ravel().tolist(),'pose_fit_rmse_px':float(np.sqrt(np.mean(np.sum((projected-pts[pose_ids])**2,axis=1))))})
    boundary=[];matrices=[]
    for j in range(len(ids)):
        A=np.array([row for P,n in zip(Ps,norm) for row in (n[j,0]*P[2]-P[0],n[j,1]*P[2]-P[1])])
        _,s,vh=np.linalg.svd(A);q=vh[-1];X=q[:3]/q[3]
        assert all((P@np.r_[X,1])[2]>0 for P in Ps)
        boundary.append(X);matrices.append({'A':A.tolist(),'singular_values':s.tolist(),'homogeneous_solution':q.tolist(),'xyz':X.tolist()})
    X=np.array(boundary);plane_mean=X.mean(0);_,_,vh=np.linalg.svd(X-plane_mean);normal=vh[-1]
    errors=[]
    for i,(name,P) in enumerate(zip(views,Ps)):
        projected=cv2.projectPoints(X,cv2.Rodrigues(P[:,:3])[0],P[:,3],K,dist)[0].reshape(-1,2)
        errors.append(float(np.sqrt(np.mean(np.sum((projected-pix[i])**2,axis=1)))))
        im=cv2.imread(str(D/name))
        for pt in corners[name][pose_ids]:cv2.circle(im,tuple(np.rint(pt).astype(int)),3,(255,150,0),1)
        cv2.polylines(im,[np.rint(pix[i]).astype(np.int32)],True,(0,200,0),2)
        for j,(p,q) in enumerate(zip(pix[i],projected)):
            cv2.circle(im,tuple(np.rint(q).astype(int)),2,(0,0,255),-1);cv2.putText(im,str(j),tuple(np.rint(p+4).astype(int)),0,.4,(0,0,255),1)
        cv2.putText(im,name+' | cyan pose, green measured, red projected',(8,20),0,.4,(255,255,255),1)
        cv2.imwrite(str(O/f'reconstruction_view_{i+1}.png'),im)
    target=np.column_stack([xy,np.zeros(len(xy))]);target_error=np.linalg.norm(X-target,axis=1)
    perimeter=np.linalg.norm(np.roll(X,-1,axis=0)-X,axis=1).sum()
    f=plt.figure(figsize=(11,5));a=f.add_subplot(121)
    closed=np.r_[X[:,:2],X[:1,:2]];truth=np.r_[target[:,:2],target[:1,:2]]
    a.plot(truth[:,0],truth[:,1],'--',label='nominal grid');a.plot(closed[:,0],closed[:,1],'-o',label='triangulated')
    for j,p in enumerate(X):a.annotate(str(j),p[:2])
    a.set(xlabel='X (board squares)',ylabel='Y (board squares)',title='Withheld interior polygon');a.axis('equal');a.legend();a.grid(alpha=.2)
    a=f.add_subplot(122,projection='3d');a.plot(X[:,0],X[:,1],X[:,2],'go-');centers=np.array([p['camera_center_board_squares'] for p in poses]);a.scatter(*centers.T,c='royalblue',marker='^',s=50)
    for j,C in enumerate(centers):a.text(*C,f'camera {j+1}')
    a.set(xlabel='X (squares)',ylabel='Y (squares)',zlabel='Z (squares)',title='Camera centers relative to the moving board')
    f.tight_layout();f.savefig(O/'boundary_and_cameras.png',dpi=150);plt.close(f)
    np.savetxt(O/'boundary_xyz_board_squares.csv',X,delimiter=',',header='X_board_squares,Y_board_squares,Z_board_squares',comments='')
    metrics={'method':'Fiducial-assisted four-view structure reconstruction, not unconstrained image-only SfM','units':'one printed chessboard square; physical square size unknown','calibration_training_images':train,'failed_corner_detection':rejected,'calibration_rmse_px':rms,'K':K.tolist(),'distortion':dist.ravel().tolist(),'poses':poses,'withheld_boundary_corner_indices':ids.tolist(),'pose_corner_indices':pose_ids.tolist(),'boundary_image_coordinates':np.array(pix).tolist(),'boundary_xyz':X.tolist(),'reprojection_rmse_px_per_view':errors,'planarity_rms_squares':float(np.sqrt(np.mean(((X-plane_mean)@normal)**2))),'nominal_grid_error_rms_squares':float(np.sqrt(np.mean(target_error**2))),'perimeter_squares':float(perimeter),'nominal_perimeter_squares':14,'first_point_DLT_worked_example':matrices[0],'limitations':['Public source images, not student captures.','The board moves; all poses are relative to its coordinate frame.','Grid pose references constrain shape and scale; nominal-grid error is not independent metrology.','Boundary is an interior rectangular region, not the outer wooden board.','No measured square length, so no distances in metres or centimetres.']}
    js(O/'reconstruction_metrics.json',metrics)
    print(json.dumps({k:metrics[k] for k in ['calibration_rmse_px','reprojection_rmse_px_per_view','planarity_rms_squares','nominal_grid_error_rms_squares','perimeter_squares']},indent=2))

def main():
    cv2.setNumThreads(4)
    if '--reconstruct-only' not in sys.argv:prepare_clips()
    reconstruct()
if __name__=='__main__':main()
