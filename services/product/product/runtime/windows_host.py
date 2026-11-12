"""Start an independent Windows runtime task with credentials transferred in memory."""
import argparse
import json
import os
from pathlib import Path
import secrets
import sys
import threading


def contain_child(process):
    """Close-on-host-death ownership for this runtime's OpenCode process tree."""
    import win32job
    handle=win32job.CreateJobObject(None,'')
    try:
        limits=win32job.QueryInformationJobObject(handle,win32job.JobObjectExtendedLimitInformation)
        limits['BasicLimitInformation']['LimitFlags']|=win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        win32job.SetInformationJobObject(handle,win32job.JobObjectExtendedLimitInformation,limits)
        win32job.AssignProcessToJobObject(handle,int(process._handle))
        return handle
    except BaseException:
        handle.Close()
        process.terminate()
        process.wait(timeout=15)
        raise


def receive(pipe_name, root):
    import win32file
    import win32pipe
    win32pipe.WaitNamedPipe(pipe_name,30000)
    handle=win32file.CreateFile(pipe_name,win32file.GENERIC_READ,0,None,win32file.OPEN_EXISTING,0,None)
    try:
        _,data=win32file.ReadFile(handle,8192)
        configuration=json.loads(data.decode('utf8'))
    finally:
        win32file.CloseHandle(handle)
    app=root/'app'
    os.environ.update({'IMAGE2PPT_PRODUCT_TOKEN':configuration['business_token'],
        'IMAGE2PPT_OFFICE_WORKER_TOKEN':configuration['office_token'],
        'IMAGE2PPT_OFFICE_MODE':'remote',
        'IMAGE2PPT_NODE':configuration['node'],
        'IMAGE2PPT_OPENCODE_BIN':str(app/'opencode.exe')})
    if configuration.get('agent_provider')=='opencode_go':
        os.environ.update(IMAGE2PPT_AGENT_PROVIDER='opencode_go',
            IMAGE2PPT_AGENT_API_KEY=configuration['agent_api_key'],
            IMAGE2PPT_AGENT_MODEL=configuration.get('agent_model','qwen3.8-flash'))
    else:
        os.environ['MOMA_API_KEY']=configuration['moma']
    sys.path.insert(0,str(app/'services/product'))
    log=(root/'business.log').open('a',encoding='utf8',buffering=1)
    sys.stdout=log;sys.stderr=log
    from product.runtime.launch import main
    sys.argv=['runtime','--root',str(root/'jobs'),'--port','8772','--opencode-port','14100']
    main()


def boot(root):
    import win32api
    import win32con
    import win32file
    import win32pipe
    import win32security
    import win32com.client
    import pywintypes
    configuration=json.load(sys.stdin)
    configuration['node']='F:/code/image2ppt/.runtime/bundle/node.exe'
    token=win32security.OpenProcessToken(win32api.GetCurrentProcess(),win32con.TOKEN_QUERY)
    sid=win32security.GetTokenInformation(token,win32security.TokenUser)[0]
    user=win32security.LookupAccountSid(None,sid)
    username=user[1]+'\\'+user[0]
    descriptor=win32security.SECURITY_DESCRIPTOR()
    acl=win32security.ACL()
    acl.AddAccessAllowedAce(win32security.ACL_REVISION,win32con.GENERIC_ALL,sid)
    descriptor.SetSecurityDescriptorDacl(1,acl,0)
    attributes=pywintypes.SECURITY_ATTRIBUTES()
    attributes.SECURITY_DESCRIPTOR=descriptor
    nonce=secrets.token_hex(12)
    pipe_name='\\\\.\\pipe\\image2ppt-runtime-'+nonce
    pipe=win32pipe.CreateNamedPipe(pipe_name,win32pipe.PIPE_ACCESS_OUTBOUND,
        win32pipe.PIPE_TYPE_MESSAGE|win32pipe.PIPE_READMODE_MESSAGE|win32pipe.PIPE_WAIT,1,8192,8192,30000,attributes)
    scheduler=win32com.client.Dispatch('Schedule.Service');scheduler.Connect()
    definition=scheduler.NewTask(0)
    definition.RegistrationInfo.Description='Image2PPT isolated OpenCode runtime'
    definition.Principal.UserId=username
    definition.Principal.LogonType=3
    definition.Settings.Enabled=True
    definition.Settings.Hidden=True
    definition.Settings.ExecutionTimeLimit='PT0S'
    definition.Settings.DisallowStartIfOnBatteries=False
    definition.Settings.StopIfGoingOnBatteries=False
    action=definition.Actions.Create(0)
    action.Path=str(root/'venv/Scripts/pythonw.exe')
    action.Arguments='"'+str(Path(__file__).resolve())+'" --root "'+str(root)+'" --receive "'+pipe_name+'"'
    action.WorkingDirectory=str(root/'app')
    name='Image2PPT-OpenCode-'+nonce[:8]
    task=scheduler.GetFolder('\\').RegisterTaskDefinition(name,definition,6,username,None,3)
    task.Run(None)
    errors=[]
    finished=threading.Event()
    def transfer():
        try:
            try:win32pipe.ConnectNamedPipe(pipe,None)
            except pywintypes.error as exc:
                if exc.winerror!=535:raise
            win32file.WriteFile(pipe,json.dumps(configuration).encode('utf8'))
            win32file.FlushFileBuffers(pipe)
        except Exception as exc:errors.append(type(exc).__name__)
        finally:finished.set()
    thread=threading.Thread(target=transfer,daemon=True);thread.start()
    try:
        if not finished.wait(45):raise TimeoutError('runtime task did not receive its bootstrap configuration')
        if errors:raise RuntimeError('runtime bootstrap '+errors[0])
    finally:
        win32file.CloseHandle(pipe)
    (root/'task.json').write_text(json.dumps({'task':name,'ports':[8772,14100]}),encoding='utf8')
    print(json.dumps({'task':name,'port':8772,'opencode_port':14100}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',required=True)
    parser.add_argument('--receive')
    options=parser.parse_args()
    root=Path(options.root).resolve()
    if options.receive:receive(options.receive,root)
    else:boot(root)
