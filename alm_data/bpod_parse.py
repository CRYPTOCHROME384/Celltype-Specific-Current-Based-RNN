# parses Bpod matlab file 

import sys, tifffile as tf, os, numpy as np, scipy.io as spio, csv, datetime
from time import sleep
from glob import glob

bsub = 'bsub -q gpu_tesla -gpu \"num=1\" '
bsub = 'bsub -q short -W 59 '
def output_keys(input=""): # structure of bpod.npy list
    d = {x:i for i,x in enumerate(['output', 'SI_files', 'vid_files', 'p1_licks', 'p2_licks', 'MATLABStartTimes'])}
    return d[input] if input else d

#0  outcome (1-5)
#1  trial type
#2  early sample binary
#3  early delay binary
#4  wrong puff binary
#5  AutoWater binary
#6  total WVT RECEIVED (in sec)
#7  recording length
#8  objective Z
#9  laser power
#10 mode
#11 licks needed
#12 protocoltype (5 or 6 or 7)
#13 reward first lick end (taste water)
#14 reaction time from go (rel)
#15 sample start
#16 last delay
#17 go cue (answer)
#18 reward
#19 freelick type
#20 reward1
def keys(input=""):
    d = {x:i for i,x in enumerate(['outcome','trialtype','earlysample','earlydelay','puff','autowater','totalwater','length','z','power','freelick_pos','licksneeded','protocol','rfl','rt','samplestart','lastdelay','go','reward','freelicktype','reward1'])}
    return d[input] if input else d

def outcomes(input=""):
    d = {x:i+1 for i,x in enumerate(['correct','incorrect','ignore','nofollow','droppednotlick','spont'])}
    return d[input] if input else d

def freelick_outcomes(input=""):
    d = {x:i+1 for i,x in enumerate(['lick','multi_lick','multi_ignore','ignore','omit','puff','laser','whisker'])}
    return d[input] if input else d

def waitForJob(job_name):
    from subprocess import check_output
    while True:
        try:
            out = check_output(['bjobs', '-J', job_name])
            if not out: break
            elif 'LSF_INVOKE_CMD' not in os.environ: print(job_name,'job(s) running')
        except: print('bpod_parse: exception while checking', job_name)
        sleep(3)

def numpy_nd_to_dict(arr):
    out = {}
    for i, x in enumerate(arr.dtype.names):
        if x.startswith('OffState') or x.startswith('OnState') or 'Bitcode' in x or x.startswith('Tup') or (len(arr[0][i])==1 and np.isnan(arr[0][i][0][0])): continue
        out[x] = arr[0][i][0] if len(arr[0][i])==1 else arr[0][i]
    return out

def npy_to_csv(npy): # not called?????
    input = np.load(npy, allow_pickle=True)
    output = [['date','time']+keys()+['si']+['vid']*len(input[2][0])+['p1lick','p2lick']]
    for i, x in enumerate(input[0].T):
        output.append([input[5][i].strftime("%d%b%Y"),input[5][i].strftime("%H:%M:%S")]+x.round(2).tolist()+[input[1][i]]+input[2][i]+[input[3][i]]+[input[1][4]])
    return output

if __name__ == "__main__":
    nchan = 2
    if len(sys.argv) == 1:
        unprocessed = []
        for d in glob('/nrs/svoboda/wangt/2pdata/*twNew*/'):
            cmd = bsub + ' -P svoboda -o '+d+'logs/parse.log python '+__file__+' '+d
            if not os.path.exists(d + 'bpod.npy'): unprocessed.append(cmd)
            else: print(cmd + ' PROCESSED')
        for p in unprocessed: print(p)
        quit()

    directory = sys.argv[1].replace('\\','/')
    if not directory.endswith('/'): directory += '/'
    os.makedirs(directory+'/bad/', exist_ok=True)
    os.makedirs(directory+'/info/', exist_ok=True)
    os.makedirs(directory+'/logs/', exist_ok=True)
    bpod_only = True if len(sys.argv)==3 else False
    print('bpod_parse: bpod only', bpod_only)
    m = glob(directory+'*.mat')
    if os.path.exists(directory+'combinedMat.mat'):
        m = directory+'combinedMat.mat'
        combined = True
    elif len(m) == 1:
        m = m[0] # assumes only 1 .mat (you place in there..) 
        combined = False
    else:
        print('bpod_parse:', len(m), 'bpod .mat files found:')
        if m: print('bpod_parse:', m)
        quit()

    ### OPEN BPOD file
    print('bpod_parse: opening', m)
    mat_f = spio.loadmat(m, chars_as_strings=True)['SessionData'][0][0] # alot of extra zeros because 2d arrays are returned

    job = directory.split('/')[-2]
    cmd = bsub + ' -P svoboda -J '+job+' -o '+directory+'logs/check_2p.log -n 4 python check_2p.py '+directory # only metadata reading
    print('bpod_parse:', cmd)
    if not bpod_only: os.system(cmd)

    vid_root = '/nrs/svoboda/wangt/videos/'
    angles = ['side', 'bottom', 'body']
    for angle in angles:
        cmd = bsub + ' -P svoboda -o '+directory+'logs/check_vid.log -n 10 python check_videos.py '+directory.replace('2pdata','videos')+angle+'/'
        print('bpod_parse:', cmd)
        if not bpod_only: os.system(cmd)

    ntrials = mat_f['nTrials'][0][0]
    output = np.zeros((21,ntrials), dtype=np.float32)

    recordings = 0
    videos = 0

    lick_timeout = 10 # could change!!!
    SI_files   = [''] * ntrials
    vid_files  = [[''] * len(angles)] * ntrials
    p1_licks   = [[]] * ntrials
    p2_licks   = [[]] * ntrials
    bpod_times = [0] * ntrials

    if combined:
        vid_idx, vid_counts = [], {}
        sessions = np.unique([x[0] for x in mat_f['OriginalMAT'][0]])
        for i, (a, b) in enumerate(np.array([[x for x in mat_f['OriginalTrial'][0]],[x[0].startswith('armed') for x in mat_f['ScanImageInfo'][0]]]).T.tolist()):
            if a not in vid_counts: vid_counts[a] = 0
            if b: vid_counts[a] += 1
            vid_idx.append(vid_counts[a])
    for i in range(ntrials):
        trial_events = [x[0] for x in mat_f['RawData'][0][0]['OriginalStateNamesByNumber'][0][i][0][mat_f['RawData'][0][0]['OriginalStateData'][0][i]-1][0]]
        trial_timings = mat_f['RawEvents'][0][0]['Trial'][0][i][0]
        trial_event_timings = numpy_nd_to_dict(trial_timings['Events'][0][0])
        trial_state_timings = numpy_nd_to_dict(trial_timings['States'][0][0])
        SI = mat_f['ScanImageInfo'][0][i][0]
        GUI = mat_f['TrialSettings'][0]['GUI'][i]
        if 'RewardFirstLick' in trial_events:
            output[keys('outcome'),i] = outcomes('correct')
            if trial_state_timings['RewardFirstLick'][-1] in trial_timings['Events'][0][0]['Tup'][0]:
                print('bpod_parse:', i, 'lick timed out', GUI['LicksNeeded'][0][0][0][0],'licks were needed autowater:', 1 if 'GiveDrop' in trial_events else 0)
                output[keys('outcome'),i] = outcomes('droppednotlick')# still don't know if touched water after this..
            if round(np.diff(trial_state_timings['RewardFirstLick'])[0]) >= lick_timeout: print(i,'lick timed out')
        elif 'Reward1' in trial_events:
            print('bpod_parse:', i, 'correct but didn\'t follow through', GUI['LicksNeeded'][0][0][0][0], 'licks were needed autowater:', 1 if 'GiveDrop' in trial_events else 0)
            output[keys('outcome'),i] = outcomes('nofollow') # pick correct but didn't follow through -- typically because of multi lick requirement
        elif 'NoResponse' in trial_events: output[keys('outcome'), i] = outcomes('ignore')
        elif 'TimeOut' in trial_events: output[keys('outcome'), i] = outcomes('incorrect')
        elif 'ProtocolType' not in GUI.dtype.fields: output[keys('outcome'), i] = outcomes('error') # why is this happening!!!
        elif GUI['ProtocolType'] == 7: output[keys('outcome'), i] = outcomes('spont')
        else: print('Error!!!!!!!!!!!!!!!!!!!!! ' + str(trial_events)) # still happens VERY rarely
        
        x = mat_f['MATLABStartTimes'][0][i]
        ix = int(x)
        dt = datetime.date.fromordinal(ix-366) #weird correction
        remainder = float(x) - ix
        hour, remainder = divmod(24 * remainder, 1)
        minute, remainder = divmod(60 * remainder, 1)
        second, remainder = divmod(60 * remainder, 1)
        microsecond = int(1e6 * remainder)
        bpod_times[i] = datetime.datetime(dt.year, dt.month, dt.day, int(hour), int(minute), int(second), microsecond) # is this substraction correct???
        output[keys('protocol'),i] = GUI['ProtocolType'][0][0][0][0] ## if 'ProtocolType' in GUI.dtype.names else 6 ## WEIRD CHANGE
        ####if i == 307: output[keys('protocol'),i] = 5
        output[keys('trialtype'),i] = mat_f['TrialTypes'][0][i]
        output[keys('earlysample'),i] = 1 if 'EarlyLickSample' in trial_events else 0 # usually aborted
        output[keys('earlysample'),i] = 1 if 'EarlyLick' in trial_events else output[2,i] # for Free Licking period
        output[keys('earlydelay'),i] = 1 if 'EarlyLickDelay'  in trial_events else 0
        output[keys('puff'),i] = 1 if 'TimeOut1' in trial_events else 0
        output[keys('autowater'),i] = 1 if 'GiveDrop' in trial_events else 0
        output[keys('totalwater'),i] = 0 if 'Reward' not in trial_state_timings else round(np.diff(trial_state_timings['Reward'])[0],4)
        output[keys('totalwater'),i] +=0 if 'GiveDrop' not in trial_state_timings else round(np.diff(trial_state_timings['GiveDrop'])[0],4)
        if 'GlobalTimer1_Start' in trial_event_timings:
            output[keys('length'),i] = trial_event_timings['GlobalTimer1_End'][0]-trial_event_timings['GlobalTimer1_Start'][0]
            if trial_event_timings['GlobalTimer1_Start'][0] != 0.0001: print('bpod_parse:', i, 'weird timer start!', trial_event_timings['GlobalTimer1_Start'])
        if SI.startswith('armed'): # not very robust...
            output[keys('z'),i] = int(float(SI.split('=')[1].split(',')[0]))
            output[keys('power'),i] = float(SI.split('=')[2].split(' ')[0])
            recording_fn = SI.split('=')[3].split(' ')[0].split('\\')[-1].strip()
            if os.path.exists(directory+recording_fn): rec_fn = directory+recording_fn
            elif len(glob(directory+'*/'+recording_fn)) == 1: rec_fn = glob(directory+'*/'+recording_fn)[0]
            elif len(glob(directory+'*/onechan/'+recording_fn)) == 1: rec_fn = glob(directory+'*/onechan/'+recording_fn)[0]
            elif recording_fn.startswith('spont_0000') and len(glob(directory+'spont_0000*.tif')): rec_fn = glob(directory+'spont_0000*.tif')[0]
            elif recording_fn.startswith('spont_0000') and len(glob(directory+'*/spont_0000*.tif')): rec_fn = glob(directory+'*/spont_0000*.tif')[0]
            else: quit('bpod_parse: 2p recording NOT FOUND: ' + recording_fn)
            if 'bad' not in rec_fn: SI_files[i] = os.path.basename(rec_fn)
            recordings += 1
            mat = mat_f['OriginalMAT'][0][0][0] if combined else os.path.basename(m) # assume first MAT has original files
            trial_num = mat_f['OriginalTrial'][0][i] if combined else i+1
            cam_files = [''] * len(angles)
            for cam_idx, angle in enumerate(angles):
                vid_base = vid_root+mat[:-4]+'/'+angle+'/'
                ## vid_wc = mat.split('_')[0]+'_'+angle+'_trial'+str(trial_num)+'_date*' # don't use this because maybe animal name is wrong...
                vid_wc = 'kd*_'+angle+'_trial'+str(trial_num)+'_date*'
                vid_fn = vid_base+'compressed/'+vid_wc+'mp4' if os.path.exists(vid_base+'compressed/') else vid_base+vid_wc+'avi'
                vid = glob(vid_fn)
                vid = sorted(vid, key = lambda fn: fn.split('_date_')[1]) # sort by DATE ecause maybe animal name is wrong...
                if combined:
                    if vid: vid = [vid[vid_idx[i]-1]]
                    else: vid = []
                if len(vid) != 1:
                    print('bpod_parse: VIDEO NOT FOUND: ' + vid_fn)
                    cam_files[cam_idx] = os.path.basename(vid_fn)
                else:
                    videos += 1               
                    cam_files[cam_idx] = os.path.basename(vid[0])
            vid_files[i] = cam_files
        output[keys('freelick_pos'),i] = GUI['RxCenter_motor_pos'][0][0][0][0] if 'RxCenter_motor_pos' in GUI else 0 # old free water
        output[keys('licksneeded'),i] = mat_f['LicksNeeded'][0][i]
        rt1, rt2 = 10, 10
        cue_event = 'AnswerPeriod' #cue_event = 'Reward1' if output[keys('protocol'),i] == 6 else 'AnswerPeriod' # old
        if 'Port1In' in trial_event_timings and cue_event in trial_state_timings:
            rt1 = trial_event_timings['Port1In'] - trial_state_timings[cue_event][0]
            rt1 = min(rt1[rt1>0]) if any(rt1>0) else 10
        if 'Port2In' in trial_event_timings and cue_event in trial_state_timings:
            rt2 = trial_event_timings['Port2In'] - trial_state_timings[cue_event][0]
            rt2 = min(rt2[rt2>0]) if any(rt2>0) else 10
        output[keys('rt'),i] = round(min(rt1, rt2), 5)
        output[keys('samplestart'),i] = trial_state_timings['SampleOn1'][0] if trial_state_timings['SampleOn1'].ndim == 1 else trial_state_timings['SampleOn1'][-1][0] # last sample start (should be only one)
        if 'DelayPeriod' in trial_state_timings: output[keys('lastdelay'),i] = trial_state_timings['DelayPeriod'][0] if trial_state_timings['DelayPeriod'].ndim == 1 else trial_state_timings['DelayPeriod'][-1][0]
        if 'AnswerPeriod' in trial_state_timings: output[keys('go'),i] = trial_state_timings['AnswerPeriod'][0]
        if 'Reward' in trial_state_timings: output[keys('reward'),i] = trial_state_timings['Reward'][0]
        if 'RewardFirstLick' in trial_state_timings: output[keys('rfl'),i] = trial_state_timings['RewardFirstLick'][1]
        if output[keys('protocol'),i] == 6:
            if 'Reward1' in trial_state_timings and 'Reward' not in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('multi_ignore')
            elif 'Reward1' in trial_state_timings and 'Reward' in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('multi_lick')
            elif 'Reward' in trial_state_timings and 'StopLicking' not in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('ignore')
            elif 'Reward' in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('lick')
            elif 'RewardOmit' in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('omit')
            elif 'RewardPuff' in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('puff')
            elif 'RewardLaser' in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('laser')
            elif 'RewardWhisker' in trial_state_timings:
                output[keys('freelicktype'),i] = freelick_outcomes('whisker')
            elif 'NoResponse' in trial_state_timings: #Early                
                output[keys('earlysample'),i] = 1
                print('bpod_parse:', i, 'no response!!!')
            else: quit('CASE UNCLEAR FOR FREE LICK!!!!!! ' + str(trial_state_timings) + ' ' + str(trial_event_timings))
        else: output[keys('freelicktype'),i] = 0
        if 'Reward1' in trial_state_timings: output[keys('reward1'),i] = trial_state_timings['Reward1'][1]
        if 'Port1In' in trial_event_timings: p1_licks[i] = np.round(trial_event_timings['Port1In'],3).tolist()
        if 'Port2In' in trial_event_timings: p2_licks[i] = np.round(trial_event_timings['Port2In'],3).tolist()

    print('bpod_parse:', recordings, '2p recordings found', videos, 'videos found')

    if recordings*len(angles) != videos:
        print('bpod_parse: MISMATCH!!!!!')
        print('bpod_parse: continuing because this is happening fairly regularly')
        print('bpod_parse: probably should figure this out...')
    
    a, b = np.unique([x for x in SI_files if x], return_counts=True)
    if np.any(b>1): 
        print('bpod_parse: duplicate 2p TIFFs found!', a[b>1], b[b>1])
        for x in np.where(SI_files == a[b>1])[0][:-1]: SI_files[x]= '' # remove all but last

    Zs = np.unique(output[keys('z')][np.where([x.startswith('trial_') for x in SI_files])[0]])
    
    np.save(directory+'bpod.npy', np.array([output, SI_files, vid_files, p1_licks, p2_licks, bpod_times]))
    
    if not bpod_only:
        if len(glob(directory+'*Motion*csv')):
            os.makedirs(directory+'Motion/', exist_ok=True)
            cmd = 'mv '+' '.join(glob(directory+'*Motion*csv'))+' '+directory+'Motion/'
            print('bpod_parse: moving motion CSVs and combining them')
            os.system(cmd)    
            out = []
            for fn in glob(directory+'Motion/*csv'):
                with open(fn, newline = '') as ww:
                    r = csv.reader(ww)
                    x = [y for y in r][1:]
                    for y in x:
                        row = [fn, int(float(y[3]))]
                        for z in y[4:7]:
                            z = z.replace('[','').replace(']','').split(' ')
                            row += list(np.array(z)[[bool(x) for x in z]].astype(float))
                        out.append(row)                
            g = open(directory+'info/motion.txt', 'w', newline='')
            w = csv.writer(g,dialect='excel-tab')
            w.writerows(out)
            g.close()
        for fn in glob(directory+'slice_00*tif'):
            if 'reshape' in fn: continue
            I = tf.imread(fn)
            I = I.reshape([int(I.shape[0]/nchan), nchan, I.shape[1], I.shape[2]])
            tf.imsave(fn[:-3]+'reshape.tif', I, imagej=True, metadata={'axes': 'ZCYX', 'Composite mode': 'composite'})
        waitForJob(job) # can't move TIFFs when doing tiff check
        
        if any([os.path.exists(directory+fn) for fn in SI_files if fn]):
            for Z in Zs:
                os.makedirs(directory+str(int(Z))+'/', exist_ok=True)
                cmd = 'mv '+' '.join([directory+x for x in np.array(SI_files)[output[keys('z')]==Z] if x!='spont_00001_00001.tif'])+' '+directory+str(int(Z))+'/'
                print('bpod_parse: moving TIFFs from planes into different folders')
                os.system(cmd)
                cmd = bsub + ' -P svoboda -J '+job+' -n 10 -o '+directory+'logs/bidi.log python bidi.py '+directory+str(int(Z))+'/'
                print('bpod_parse:', cmd)
                os.system(cmd)
            spont = glob(directory+'spont_00*_*.tif')
            if len(spont):
                os.makedirs(directory+'spont/', exist_ok=True)
                cmd = 'mv '+' '.join(spont)+' '+directory+'spont/'
                print('bpod_parse: moving TIFFs from planes into different folders')
                os.system(cmd)
                cmd = bsub + ' -P svoboda -J '+job+' -n 10 -o '+directory+'logs/bidi.log python bidi.py '+directory+'spont/'
                print('bpod_parse:', cmd)
                os.system(cmd)
            for Z in Zs:
                cmd = bsub + ' -P svoboda -J '+job+' -n 10 -o '+directory+'logs/avg.log python avg_trials.py '+directory+str(int(Z))+'/'
                print('bpod_parse:', cmd)
                os.system(cmd)
            if os.path.exists(directory+'spont/'):
                cmd = bsub + ' -P svoboda -J '+job+' -n 10 -o '+directory+'logs/avg.log python avg_trials.py '+directory+'spont/'
                print('bpod_parse:', cmd)
                os.system(cmd)
            waitForJob(job)
            print('bpod_parse: delete any unnecessary bidi images or move spont images before averaging. then RUN:')
            print('bpod_parse: bsub -P svoboda -o '+directory+'process.log python '+__file__+' '+directory)
            s2p_out = '<BR>'.join(['python run_suite2p.py '+directory+str(int(Z))+'/' for Z in Zs]+['python run_suite2p.py '+directory+'spont/','python session_analysis.py '+directory])
            try:
                print('emailing...')
                from mailjet_rest import Client
                mailjet = Client(auth=('78055a65c13ef638d5a8b5fe05c503f6', '4fee843e5bdba74c0c5bbebc3d35a1ba'), version='v3.1')
                data = {
                  'Messages': [ {
                      "From": { "Email": "timtimspim@gmail.com", "Name": "Light" },
                      "To": [ { "Email": "wangt@janelia.hhmi.org",  "Name": "Tim" } ],
                      "Subject": os.path.basename(directory[:-1])+' bidi and averaging done',
                      "HTMLPart": s2p_out,
                   } ]
                }
                result = mailjet.send.create(data=data)
            except: print('mail fail')
            # quit()
        else:
            for Z in Zs:
                cmd = bsub + ' -P svoboda -J '+job+' -n 10 -o '+directory+'logs/avg.log python avg_trials.py '+directory+str(int(Z))+'/'
                print('bpod_parse:', cmd)
                os.system(cmd)
            if os.path.exists(directory+'spont/'):
                cmd = bsub + ' -P svoboda -J '+job+' -n 10 -o '+directory+'logs/avg.log python avg_trials.py '+directory+'spont/'
                print('bpod_parse:', cmd)
                os.system(cmd)
    else: waitForJob(job) # we still need to wait for TIFF checks to complete

    cmd = bsub + ' -P svoboda -J '+job+' -o '+directory+'logs/check_mm.log python check_mismatches.py '+directory
    print('bpod_parse:', cmd)
    if not bpod_only: os.system(cmd)
    waitForJob(job)
    cmd = bsub + ' -P svoboda -J '+job+' -o '+directory+'logs/check_sess.log python session_analysis.py '+directory
    print('bpod_parse:', cmd)
    if not bpod_only: os.system(cmd)
    
    print('bpod_parse: check summaries. when ready, remove bad planes and run:')
    for Z in Zs: print('python run_suite2p.py '+directory+str(int(Z))+'/')
    if os.path.exists(directory+'spont/'): print('python run_suite2p.py '+directory+'spont/')
    # print('bpod_parse: or if suite2p is already done, run:')
    # for Z in Zs: print(bsub + ' -P svoboda -o /dev/null -n 25 python trial_analysis.Jan15.py '+directory+str(int(Z))+'/')
    # print('bpod_parse: or if suite2p is already done, run:')
    # for Z in Zs: print(bsub + ' -P svoboda -o /dev/null -n 25 python free_trial_analysis.py '+directory+str(int(Z))+'/')
