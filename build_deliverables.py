from pathlib import Path
import json, csv, subprocess, platform, importlib.metadata, hashlib, shutil, sys, html, zipfile
import numpy as np, cv2, imageio_ffmpeg
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import cv_course

R=Path(__file__).resolve().parent;O=R/'results';D=R/'data'
M=json.loads((O/'reconstruction_metrics.json').read_text());F=json.loads((O/'flow/summary.json').read_text())
URL='https://github.com/aabdelkarem/csc8830-module6-motion-reconstruction'
styles=getSampleStyleSheet();styles['Normal'].fontSize=10;styles['Normal'].leading=13
styles['Heading1'].textColor=colors.HexColor('#17486A');styles['Code'].fontSize=8.5;styles['Code'].leading=11
story=[]
def p(t):story.extend([Paragraph(t,styles['Normal']),Spacer(1,7)])
def h(t):story.append(Paragraph(t,styles['Heading1']))
def code(t):story.extend([Preformatted(t,styles['Code']),Spacer(1,7)])
def pic(path,w=510):
    im=Image(str(path));original_width,original_height=im.imageWidth,im.imageHeight
    im.drawWidth=w;im.drawHeight=w*original_height/original_width;story.extend([im,Spacer(1,7)])
def table(rows,widths):
    t=Table(rows,colWidths=widths,repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#17486A')),('TEXTCOLOR',(0,0),(-1,0),colors.white),('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),('FONTSIZE',(0,0),(-1,-1),8.5),('GRID',(0,0),(-1,-1),.3,colors.HexColor('#B0C0CC')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F1F5F8')]),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,10)])
def page():story.append(PageBreak())
def footer(c,d):c.setFont('Helvetica',8);c.setFillColor(colors.grey);c.drawString(42,25,'Aya Abdelkarem | CSc 8830 | Actual public-data experiment');c.drawRightString(570,25,str(d.page))

h('Module 6: motion and four-view reconstruction')
p('<b>Aya Abdelkarem - CSc 8830, Spring 2026 makeup work</b>')
p('Repository: <link href="'+URL+'" color="#17486A">'+URL+'</link>')
p('<b>Scope.</b> Dense optical flow was computed on two real 30-second video excerpts. Six visually annotated points per video were checked with pyramidal Lucas-Kanade and a separate small-motion implementation. Four real photographs were used to reconstruct a held-out planar polygon and estimate its boundary.')
p('<b>Provenance and limits.</b> All imagery comes from cited public examples, not student captures. The reconstruction uses known chessboard points to estimate camera pose: it is a fiducial-assisted structure-reconstruction example, not unrestricted image-only SfM. The boundary is an interior rectangular board region. Distances are in board-square units because no physical square length is documented. This package does not claim approval of a substitute capture or live demonstration.')
table([['Executed item','Evidence'],['Pedestrian video','30.00 s; 300 frames; native 10 fps'],['Violin video','30.00 s; 750 frames; native 25 fps'],['Frame-pair validation','6 visually annotated points per sample'],['Four-view reconstruction','4 images, 8 held-out boundary points'],['Calibration','7 separate images; RMSE %.3f px'%M['calibration_rmse_px']],['Demonstration','Module_6_Working_Demo.mp4; actual outputs']], [160,350])
p('Video excerpts are resized to 640 pixels wide without frame-rate conversion: vtest.avi seconds 10-40 and the music video seconds 45-75. Validation uses consecutive frames at approximately source times 15.0/15.1 seconds and 50.00/50.04 seconds. Frames at relative indices 50/51 and 125/126 are paired; coordinates use the resized images.')
pic(O/'flow/video_1/flow_sample.png')
p('Example output: original frame at left, dense flow at right. Hue represents direction; brightness represents magnitude on a fixed 0-8 px/frame scale. Black background is near-zero apparent motion.')
page();h('1. Tracking derivation and bilinear interpolation')
p('Brightness constancy assumes I(x,y,t) = I(x+u,y+v,t+dt). First-order expansion gives Ix*u + Iy*v + It = 0. One equation cannot determine two displacement components (the aperture problem). Lucas-Kanade assumes a common displacement in a textured neighborhood and stacks its pixel equations.')
code('For each pixel i in an 11 x 11 patch:\nG_i = [Ix_i, Iy_i],  b_i = I1_i - I2_i\nMinimize ||G d - b||^2.\nNormal equations: (G^T G) d = G^T b.\nUse least squares/SVD rather than explicitly inverting G^T G.\n\nIterative refinement at displacement d:\nb_i(d) = I1(x_i,y_i) - I2(x_i+d_x,y_i+d_y)\nDelta = least_squares(G, b(d)); d <- d + Delta.\nStop at ||Delta|| < 0.001 px or after 25 iterations.')
p('Both independent implementations use the same starting points, but the scratch implementation is single-scale. OpenCV pyramidal LK uses 21x21 windows, up to three pyramid levels, and 30 iterations with 0.01-pixel tolerance. Small displacement, sufficiently stable brightness, texture in both axes, and limited occlusion are necessary for accurate tracking.')
p('Subpixel warping requires interpolation. For x=i+a, y=j+b with 0&lt;=a,b&lt;1:')
code('I(x,y) = (1-a)(1-b) I[j,i] + a(1-b) I[j,i+1]\n       + (1-a)b I[j+1,i] + ab I[j+1,i+1].\nWeights sum to 1; for a=.3, b=.6: .28, .12, .42, .18.')
calcs=[]
for i,name in enumerate(['pedestrians','violin'],1):
    item=json.loads((O/'manual_annotations.json').read_text())['annotations'][name]
    cap=cv2.VideoCapture(str(D/(name+'_30s.mp4')));cap.set(cv2.CAP_PROP_POS_FRAMES,F[i-1]['validation_pair_relative_index']);_,a=cap.read();_,b=cap.read();cap.release()
    a=cv2.cvtColor(a,cv2.COLOR_BGR2GRAY)/255.;b=cv2.cvtColor(b,cv2.COLOR_BGR2GRAY)/255.
    px,py=item['points_first'][0];oy,ox=np.mgrid[-5:6,-5:6];xx=px+ox.ravel();yy=py+oy.ravel();gy,gx=np.gradient(a)
    G=np.c_[cv_course.bilinear(gx,xx,yy),cv_course.bilinear(gy,xx,yy)];rhs=cv_course.bilinear(a,xx,yy)-cv_course.bilinear(b,xx,yy)
    gtg=G.T@G;gtb=G.T@rhs;step=np.linalg.lstsq(G,rhs,rcond=None)[0]
    intens=[a[py,px],a[py,px+1],a[py+1,px],a[py+1,px+1]];val=float(np.dot([.28,.12,.42,.18],intens))
    c={'sample':name,'point':[px,py],'GtG':gtg.tolist(),'Gtb':gtb.tolist(),'first_iteration_step_px':step.tolist(),'bilinear_intensities':intens,'interpolated_intensity':val};calcs.append(c)
    p('<b>Actual numerical example: '+name+', point 0.</b> Intensities normalized to [0,1].')
    code('G^T G = '+np.array2string(gtg,precision=6)+'\nG^T b = '+np.array2string(gtb,precision=6)+'\nFirst step = '+np.array2string(step,precision=6)+' px\nBilinear values = '+np.array2string(np.array(intens),precision=4)+'\nI(x+.3,y+.6) = %.6f'%val)
js=lambda f,x:f.write_text(json.dumps(x,indent=2));js(O/'tracking_worked_calculations.json',calcs)
for i,name in enumerate(['Pedestrians','Violin'],1):
    page();h('2.%d %s: measured tracking results'%(i,name))
    s=F[i-1];p('Processed %.2f seconds (%d frames). Mean dense-flow magnitude over all pixels and frame pairs: %.4f px/frame. Global means mix moving foreground and stationary background and are not physical speeds.'%(s['duration_seconds'],s['processed_frames'],s['mean_flow_magnitude_px_per_frame']))
    rows=list(csv.DictReader(open(O/f'flow/video_{i}/tracking_validation.csv')))
    tr=[['ID','Start (x,y)','Visual next (x,y)','PyrLK next (x,y)','LK error','Scratch error']]
    for r in rows:tr.append([r['id'],f"({float(r['x0']):.0f},{float(r['y0']):.0f})",f"({float(r['manual_x1']):.0f},{float(r['manual_y1']):.0f})",f"({float(r['pyrLK_x1']):.2f},{float(r['pyrLK_y1']):.2f})",'%.2f px'%float(r['pyrLK_error_px']),'%.2f px'%float(r['scratch_error_px'])])
    table(tr,[25,80,103,122,85,95]);p('Mean endpoint error: pyramidal LK %.3f px; scratch LK %.3f px. The tracker validity flag was true for all six points, which does not mean all locations are accurate.'%(s['pyrLK_mean_endpoint_error_px'],s['scratch_mean_endpoint_error_px']))
    pic(O/f'flow/video_{i}/tracking_green_manual_red_predicted.png',360 if i==1 else 510)
    p('Green circles mark visually estimated next-frame positions; red marks are tracker outputs. These references were recorded by computer-assisted visual inspection before running either tracker. Integer marking uncertainty is approximately +/-3 px; these are not independently measured ground-truth labels. CSV files preserve all points, including failures.')
    if i==1:p('Foreground pedestrians produce flow while much of the static pavement and grass remains dark. The largest point discrepancy occurs near the central pedestrian and sign, where occlusion and mixed patch motion undermine the common-displacement assumption. Most other discrepancies are within the approximate marking precision.')
    else:p('The bow and arm yield more motion than the furniture, with thin structures and blur complicating tracking. A nominally static keyboard-edge point drifts about 10.48 px because its tracking window overlaps the moving arm; the valid flag does not detect this semantic failure. The elbow reference is also uncertain under deformation. These failures motivate forward-backward checks, better feature placement and exclusion masks rather than claiming perfect tracking.')
page();h('3. Four-view reconstruction: camera model')
p('Images left01, left02, left06 and left07 from OpenCV form the four reconstruction views. The board moves relative to the camera; equivalently, camera poses are expressed in the board coordinate system. Seven different images calibrate intrinsics and distortion: '+', '.join(M['calibration_training_images'])+'. Corner detection failed on left04 and left05, and these were excluded.')
p('The 9x6 inner-corner grid defines reference coordinates (column,row,0) in board-square units. Pose estimation uses only its outer ring. Eight strictly interior corners define a rectangular target polygon; they are withheld from all four pose fits. Their image locations are then triangulated without supplying their target world coordinates to the DLT solver.')
code('Image model: p ~ K [R | t] X_h (after undistortion)\nK = '+np.array2string(np.array(M['K']),precision=6)+'\nDistortion [k1,k2,p1,p2,k3] =\n'+np.array2string(np.array(M['distortion']),precision=7))
p('Calibration reprojection RMSE = %.6f px. solvePnP estimates each R,t from the outer ring. A camera center in the board frame is C = -R<super>T</super>t. All translations below are in board-square units; no centimetre or metre conversion is asserted.'%M['calibration_rmse_px'])
for i,pose in enumerate(M['poses'],1):
    code('View %d, %s; pose RMSE %.6f px\nR = %s\nt = %s\nC = %s'%(i,pose['file'],pose['pose_fit_rmse_px'],np.array2string(np.array(pose['R']),precision=6),np.array2string(np.array(pose['t_board_squares']),precision=6),np.array2string(np.array(pose['camera_center_board_squares']),precision=6)))
page();h('4. Four-view DLT: complete numerical example')
p('For each undistorted normalized observation (x_i,y_i) and projection matrix P_i=[R_i|t_i], the relation (x_i,y_i,1) cross (P_i X_h)=0 supplies two independent equations. Stack all four views into an 8x4 matrix A. The right singular vector associated with the smallest singular value minimizes ||A X_h|| subject to ||X_h||=1. Divide by the fourth coordinate, then check positive depth in every camera.')
code('Rows for each view: x_i P_i[2,:] - P_i[0,:]\n                    y_i P_i[2,:] - P_i[1,:]')
e=M['first_point_DLT_worked_example'];code('Actual A for boundary point 0:\n'+np.array2string(np.array(e['A']),precision=6,max_line_width=88)+'\nSingular values = '+np.array2string(np.array(e['singular_values']),precision=8)+'\nHomogeneous solution = '+np.array2string(np.array(e['homogeneous_solution']),precision=8)+'\nX = '+np.array2string(np.array(e['xyz']),precision=8))
table([['Point','X (squares)','Y (squares)','Z (squares)']]+[[str(i)]+['%.6f'%v for v in q] for i,q in enumerate(M['boundary_xyz'])],[60,150,150,150])
p('The ordered points trace the boundary. Perimeter = sum ||X_(j+1)-X_j||, including the closing segment. The fitted plane is obtained by SVD of centered 3D points; its smallest right singular vector is the plane normal.')
p('Perimeter = %.6f squares (nominal 14). Plane-fit RMS = %.6f squares. RMS difference from nominal interior grid coordinates = %.6f squares. Per-view reprojection RMSE = %s px.'%(M['perimeter_squares'],M['planarity_rms_squares'],M['nominal_grid_error_rms_squares'],', '.join('%.4f'%v for v in M['reprojection_rmse_px_per_view'])))
p('Reprojection and nominal-grid agreement are internal checks. The same planar grid provides camera-pose references, so these values are not independent real-world accuracy measurements. Planar scenes have ambiguities without calibration/reference information; known grid references resolve scale here. An unrestricted SfM pipeline would instead estimate camera motion and structure jointly from arbitrary tracked features, up to scale, and normally refine by bundle adjustment.')
page();h('5. Boundary and camera-position evidence');pic(O/'boundary_and_cameras.png')
for i in (1,2):pic(O/f'reconstruction_view_{i}.png',260)
page();h('Four-view image evidence, continued')
for i in (3,4):pic(O/f'reconstruction_view_{i}.png',360)
p('Cyan points: pose references. Green outline: measured interior boundary. Red dots: projections of the reconstructed boundary. The selected rectangle is explicitly marked in each view. Original photographs and full numeric camera matrices are included in the source package.')
page();h('6. Reproduction, references and submission map')
code('python3.12 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\npython download_sources.py\npython run_module6.py\npython build_deliverables.py')
p('Submit this PDF and Module_6_Working_Demo.mp4 to the Module 6 item, together with the repository link. Source and numeric results are in Module_6_Source_and_Results.zip. Separate full-length flow videos are also in the results archive. Nothing is automatically submitted by these programs.')
p('This package demonstrates implemented methods and actual execution. If the instructor requires personal captures or an unrestricted SfM implementation, the public-data/fiducial-assisted example does not establish that requirement. No live TA attendance or approval is claimed. Review the derivation, failures and limitations before submission.')
refs=[('OpenCV optical-flow documentation','https://docs.opencv.org/4.x/d4/dee/tutorial_optical_flow.html'),('OpenCV camera calibration','https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html'),('OpenCV calib3d: solvePnP, projectPoints and calibration','https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html'),('OpenCV 4.10.0 sample videos and photographs','https://github.com/opencv/opencv/tree/4.10.0/samples/data'),('Mobile-VideoGPT public violin sample','https://github.com/Amshaker/Mobile-VideoGPT/tree/c82a4cbc9edcb4eacd6223b4389699f24aca6635/sample_videos')]
for title,url in refs:p('<b>'+title+'</b><br/><link href="'+url+'" color="#17486A">'+url+'</link>')
p('The OpenCV source license and both Mobile-VideoGPT notices are preserved. The latter repository LICENSE says CC BY 4.0 while its README says CC BY-NC-SA 4.0; both are retained with source attribution. This is noncommercial coursework. Source URLs, file sizes and SHA-256 hashes are recorded in data/SOURCES.json. The supplied data and annotations are not independently collected or human-validated ground truth.')
SimpleDocTemplate(str(R/'Module_6_Report.pdf'),pagesize=(612,792),leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)

# Demonstration from actual executed outputs. It is not a fabricated screen recording.
if '--skip-demo' not in sys.argv:
    tmp=R/'demo_intermediate.mp4';writer=cv2.VideoWriter(str(tmp),cv2.VideoWriter_fourcc(*'mp4v'),25,(1280,720));assert writer.isOpened()
    def panel(im,title,subtitle):
        out=np.full((720,1280,3),(27,21,16),np.uint8);scale=min(1240/im.shape[1],600/im.shape[0]);im=cv2.resize(im,(round(im.shape[1]*scale),round(im.shape[0]*scale)));x=(1280-im.shape[1])//2;y=85+(600-im.shape[0])//2;out[y:y+im.shape[0],x:x+im.shape[1]]=im
        cv2.putText(out,title,(25,35),0,.8,(255,255,255),2);cv2.putText(out,subtitle,(25,66),0,.5,(170,220,230),1);return out
    for i,name in enumerate(['Pedestrians','Violin'],1):
        cap=cv2.VideoCapture(str(O/f'flow/video_{i}/optical_flow.mp4'));fps=cap.get(cv2.CAP_PROP_FPS);cached=None;current=-1
        for k in range(750):
            target=min(int(k*fps/25),int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1)
            while current<target:
                ok,cached=cap.read();assert ok;current+=1
            writer.write(panel(cached,f'{name}: 30-second optical-flow output',f'Public source, not student capture | original left / flow right | source {fps:g} fps | t={k/25:.2f} s'))
        cap.release()
    for i in range(1,5):
        im=panel(cv2.imread(str(O/f'reconstruction_view_{i}.png')),f'Four-view reconstruction: camera {i}','Known outer-grid pose references; interior polygon withheld from pose fitting')
        for _ in range(75):writer.write(im)
    im=panel(cv2.imread(str(O/'boundary_and_cameras.png')),'Reconstructed boundary and camera positions','Perimeter 14.0129 board squares | units are NOT metres | fiducial-assisted example')
    for _ in range(100):writer.write(im)
    writer.release();subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(tmp),'-c:v','libx264','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(R/'Module_6_Working_Demo.mp4')],check=True);tmp.unlink()
env={'python':platform.python_version(),'packages':{n:importlib.metadata.version(n) for n in ['numpy','scipy','opencv-python-headless','matplotlib','pillow','reportlab','imageio-ffmpeg']}}
js(R/'TESTED_ENVIRONMENT.json',env)
(R/'requirements.txt').write_text('\n'.join(k+'=='+v for k,v in env['packages'].items())+'\n')
cap=cv2.VideoCapture(str(R/'Module_6_Working_Demo.mp4'));assert cap.isOpened();duration=cap.get(cv2.CAP_PROP_FRAME_COUNT)/cap.get(cv2.CAP_PROP_FPS);assert abs(duration-76)<.05;cap.release()
js(R/'DEMO_VALIDATION.json',{'duration_seconds':duration,'expected_seconds':76,'fps':25,'resolution':[1280,720],'contents':'Two full 30-second flow sequences plus four-view and camera/boundary output visuals','source':'Actual computed outputs; not a recording of a human demonstration'})
print('Wrote report and validated',duration,'second demonstration')
