#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on Mayıs 27, 2026, at 16:49
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'untitled_blocks'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '001',
    'group': ["1" ,"2"],
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\leyla\\Desktop\\DFKI\\untitled_blocks.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1.0000, 1.0000, 1.0000], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1.0000, 1.0000, 1.0000]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('skip_welcome') is None:
        # initialise skip_welcome
        skip_welcome = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='skip_welcome',
        )
    if deviceManager.getDevice('skip_expdetails') is None:
        # initialise skip_expdetails
        skip_expdetails = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='skip_expdetails',
        )
    if deviceManager.getDevice('skip_pracinfo') is None:
        # initialise skip_pracinfo
        skip_pracinfo = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='skip_pracinfo',
        )
    if deviceManager.getDevice('practice_prog') is None:
        # initialise practice_prog
        practice_prog = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='practice_prog',
        )
    if deviceManager.getDevice('start_session') is None:
        # initialise start_session
        start_session = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='start_session',
        )
    if deviceManager.getDevice('trial_prog') is None:
        # initialise trial_prog
        trial_prog = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='trial_prog',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "welcome" ---
    welcome_text = visual.TextStim(win=win, name='welcome_text',
        text='Welcome to the Experiment!\nThank you for participating. \n\nIn this experiment, you will solve tangram puzzles. Tangram puzzles are blacked-out shapes that you recreate using the 7 differently shaped and colored small pieces. The goal is to match the outline of the blacked-out shape as closely as possible using the pieces at your disposal.\n\nIf this sounds a bit tricky, don’t worry! We’ll start with a practice session where you’ll see some examples of how tangram puzzles work. After the practice session, the actual experiment will begin.\n\nPress SPACEBAR to see the experiment details.\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    skip_welcome = keyboard.Keyboard(deviceName='skip_welcome')
    
    # --- Initialize components for Routine "exp_details" ---
    details = visual.TextStim(win=win, name='details',
        text='The experiment has 2 sessions, and in each session: (1) You will have 15 minutes to solve as many puzzles as you can, (2) The timer starts when you begin the first puzzle, (3)  After 15 minutes, the session will end automatically, (4) At the end of each session, you’ll complete a short questionnaire. Please inform the experimenter when you finish a session.\n\nImportant Notes: (1) Try to solve the puzzles CORRECTLY to the best of your ability, (2) After completing a puzzle, press SPACEBAR to move to the next one, (3) If you get stuck on a puzzle for more than 5 minutes, a message will appear saying, "Press S to skip this puzzle." You can skip or continue trying, (4) The experimenter will be in the room to answer any questions.\n\nWhen you’re ready to start the practice session, press SPACEBAR.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    skip_expdetails = keyboard.Keyboard(deviceName='skip_expdetails')
    
    # --- Initialize components for Routine "practice_info" ---
    practice_inf = visual.TextStim(win=win, name='practice_inf',
        text="Practice Session\n\nYou will see 3 different puzzles as example and some information text with them. After each of the puzzle you will see their solution. You don't have to solve all of them but you can give it a try in the second puzzle to have a sense of tangram game. \n\nPress SPACEBAR to see the puzzles. ",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    skip_pracinfo = keyboard.Keyboard(deviceName='skip_pracinfo')
    
    # --- Initialize components for Routine "practice" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='',
        font='Arial',
        pos=[0,0], draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    practice_puzzles = visual.ImageStim(
        win=win,
        name='practice_puzzles', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    practice_prog = keyboard.Keyboard(deviceName='practice_prog')
    
    # --- Initialize components for Routine "session_info" ---
    session_inf = visual.TextStim(win=win, name='session_inf',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    start_session = keyboard.Keyboard(deviceName='start_session')
    
    # --- Initialize components for Routine "trial" ---
    puzzles = visual.ImageStim(
        win=win,
        name='puzzles', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1.0000, 1.0000, 1.0000], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    trial_prog = keyboard.Keyboard(deviceName='trial_prog')
    skip_msg = visual.TextStim(win=win, name='skip_msg',
        text='press S to skip this puzzle',
        font='Arial',
        pos=(0, -0.4), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "bye" ---
    bye_txt = visual.TextStim(win=win, name='bye_txt',
        text="That's all!\n\nThank you for your participation.",
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color=[-1.0000, -1.0000, -1.0000], colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "welcome" ---
    # create an object to store info about Routine welcome
    welcome = data.Routine(
        name='welcome',
        components=[welcome_text, skip_welcome],
    )
    welcome.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for skip_welcome
    skip_welcome.keys = []
    skip_welcome.rt = []
    _skip_welcome_allKeys = []
    # store start times for welcome
    welcome.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    welcome.tStart = globalClock.getTime(format='float')
    welcome.status = STARTED
    thisExp.addData('welcome.started', welcome.tStart)
    welcome.maxDuration = None
    # keep track of which components have finished
    welcomeComponents = welcome.components
    for thisComponent in welcome.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "welcome" ---
    welcome.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *welcome_text* updates
        
        # if welcome_text is starting this frame...
        if welcome_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            welcome_text.frameNStart = frameN  # exact frame index
            welcome_text.tStart = t  # local t and not account for scr refresh
            welcome_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(welcome_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'welcome_text.started')
            # update status
            welcome_text.status = STARTED
            welcome_text.setAutoDraw(True)
        
        # if welcome_text is active this frame...
        if welcome_text.status == STARTED:
            # update params
            pass
        
        # *skip_welcome* updates
        waitOnFlip = False
        
        # if skip_welcome is starting this frame...
        if skip_welcome.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            skip_welcome.frameNStart = frameN  # exact frame index
            skip_welcome.tStart = t  # local t and not account for scr refresh
            skip_welcome.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(skip_welcome, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'skip_welcome.started')
            # update status
            skip_welcome.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(skip_welcome.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(skip_welcome.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if skip_welcome.status == STARTED and not waitOnFlip:
            theseKeys = skip_welcome.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _skip_welcome_allKeys.extend(theseKeys)
            if len(_skip_welcome_allKeys):
                skip_welcome.keys = _skip_welcome_allKeys[-1].name  # just the last key pressed
                skip_welcome.rt = _skip_welcome_allKeys[-1].rt
                skip_welcome.duration = _skip_welcome_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            welcome.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in welcome.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "welcome" ---
    for thisComponent in welcome.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for welcome
    welcome.tStop = globalClock.getTime(format='float')
    welcome.tStopRefresh = tThisFlipGlobal
    thisExp.addData('welcome.stopped', welcome.tStop)
    # check responses
    if skip_welcome.keys in ['', [], None]:  # No response was made
        skip_welcome.keys = None
    thisExp.addData('skip_welcome.keys',skip_welcome.keys)
    if skip_welcome.keys != None:  # we had a response
        thisExp.addData('skip_welcome.rt', skip_welcome.rt)
        thisExp.addData('skip_welcome.duration', skip_welcome.duration)
    thisExp.nextEntry()
    # the Routine "welcome" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "exp_details" ---
    # create an object to store info about Routine exp_details
    exp_details = data.Routine(
        name='exp_details',
        components=[details, skip_expdetails],
    )
    exp_details.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for skip_expdetails
    skip_expdetails.keys = []
    skip_expdetails.rt = []
    _skip_expdetails_allKeys = []
    # store start times for exp_details
    exp_details.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    exp_details.tStart = globalClock.getTime(format='float')
    exp_details.status = STARTED
    thisExp.addData('exp_details.started', exp_details.tStart)
    exp_details.maxDuration = None
    # keep track of which components have finished
    exp_detailsComponents = exp_details.components
    for thisComponent in exp_details.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "exp_details" ---
    exp_details.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *details* updates
        
        # if details is starting this frame...
        if details.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            details.frameNStart = frameN  # exact frame index
            details.tStart = t  # local t and not account for scr refresh
            details.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(details, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'details.started')
            # update status
            details.status = STARTED
            details.setAutoDraw(True)
        
        # if details is active this frame...
        if details.status == STARTED:
            # update params
            pass
        
        # *skip_expdetails* updates
        waitOnFlip = False
        
        # if skip_expdetails is starting this frame...
        if skip_expdetails.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            skip_expdetails.frameNStart = frameN  # exact frame index
            skip_expdetails.tStart = t  # local t and not account for scr refresh
            skip_expdetails.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(skip_expdetails, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'skip_expdetails.started')
            # update status
            skip_expdetails.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(skip_expdetails.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(skip_expdetails.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if skip_expdetails.status == STARTED and not waitOnFlip:
            theseKeys = skip_expdetails.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _skip_expdetails_allKeys.extend(theseKeys)
            if len(_skip_expdetails_allKeys):
                skip_expdetails.keys = _skip_expdetails_allKeys[-1].name  # just the last key pressed
                skip_expdetails.rt = _skip_expdetails_allKeys[-1].rt
                skip_expdetails.duration = _skip_expdetails_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            exp_details.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in exp_details.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "exp_details" ---
    for thisComponent in exp_details.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for exp_details
    exp_details.tStop = globalClock.getTime(format='float')
    exp_details.tStopRefresh = tThisFlipGlobal
    thisExp.addData('exp_details.stopped', exp_details.tStop)
    # check responses
    if skip_expdetails.keys in ['', [], None]:  # No response was made
        skip_expdetails.keys = None
    thisExp.addData('skip_expdetails.keys',skip_expdetails.keys)
    if skip_expdetails.keys != None:  # we had a response
        thisExp.addData('skip_expdetails.rt', skip_expdetails.rt)
        thisExp.addData('skip_expdetails.duration', skip_expdetails.duration)
    thisExp.nextEntry()
    # the Routine "exp_details" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_info" ---
    # create an object to store info about Routine practice_info
    practice_info = data.Routine(
        name='practice_info',
        components=[practice_inf, skip_pracinfo],
    )
    practice_info.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for skip_pracinfo
    skip_pracinfo.keys = []
    skip_pracinfo.rt = []
    _skip_pracinfo_allKeys = []
    # store start times for practice_info
    practice_info.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_info.tStart = globalClock.getTime(format='float')
    practice_info.status = STARTED
    thisExp.addData('practice_info.started', practice_info.tStart)
    practice_info.maxDuration = None
    # keep track of which components have finished
    practice_infoComponents = practice_info.components
    for thisComponent in practice_info.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_info" ---
    practice_info.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *practice_inf* updates
        
        # if practice_inf is starting this frame...
        if practice_inf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            practice_inf.frameNStart = frameN  # exact frame index
            practice_inf.tStart = t  # local t and not account for scr refresh
            practice_inf.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(practice_inf, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'practice_inf.started')
            # update status
            practice_inf.status = STARTED
            practice_inf.setAutoDraw(True)
        
        # if practice_inf is active this frame...
        if practice_inf.status == STARTED:
            # update params
            pass
        
        # *skip_pracinfo* updates
        waitOnFlip = False
        
        # if skip_pracinfo is starting this frame...
        if skip_pracinfo.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            skip_pracinfo.frameNStart = frameN  # exact frame index
            skip_pracinfo.tStart = t  # local t and not account for scr refresh
            skip_pracinfo.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(skip_pracinfo, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'skip_pracinfo.started')
            # update status
            skip_pracinfo.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(skip_pracinfo.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(skip_pracinfo.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if skip_pracinfo.status == STARTED and not waitOnFlip:
            theseKeys = skip_pracinfo.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _skip_pracinfo_allKeys.extend(theseKeys)
            if len(_skip_pracinfo_allKeys):
                skip_pracinfo.keys = _skip_pracinfo_allKeys[-1].name  # just the last key pressed
                skip_pracinfo.rt = _skip_pracinfo_allKeys[-1].rt
                skip_pracinfo.duration = _skip_pracinfo_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            practice_info.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in practice_info.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_info" ---
    for thisComponent in practice_info.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_info
    practice_info.tStop = globalClock.getTime(format='float')
    practice_info.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_info.stopped', practice_info.tStop)
    # check responses
    if skip_pracinfo.keys in ['', [], None]:  # No response was made
        skip_pracinfo.keys = None
    thisExp.addData('skip_pracinfo.keys',skip_pracinfo.keys)
    if skip_pracinfo.keys != None:  # we had a response
        thisExp.addData('skip_pracinfo.rt', skip_pracinfo.rt)
        thisExp.addData('skip_pracinfo.duration', skip_pracinfo.duration)
    thisExp.nextEntry()
    # the Routine "practice_info" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_trials = data.TrialHandler2(
        name='practice_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('C:\Users\leyla\Desktop\DFKI\AR_cognitive_load\experiment_design\practice.xlsx'), 
        seed=None, 
    )
    thisExp.addLoop(practice_trials)  # add the loop to the experiment
    thisPractice_trial = practice_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
    if thisPractice_trial != None:
        for paramName in thisPractice_trial:
            globals()[paramName] = thisPractice_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_trial in practice_trials:
        currentLoop = practice_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_trial.rgb)
        if thisPractice_trial != None:
            for paramName in thisPractice_trial:
                globals()[paramName] = thisPractice_trial[paramName]
        
        # --- Prepare to start Routine "practice" ---
        # create an object to store info about Routine practice
        practice = data.Routine(
            name='practice',
            components=[text_2, practice_puzzles, practice_prog],
        )
        practice.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text_2.setPos(msg_pos)
        text_2.setText(msg
        
        )
        practice_puzzles.setImage('DFKI/AR_cognitive_load/experiment_design/conditions/practice.xlsx')
        # create starting attributes for practice_prog
        practice_prog.keys = []
        practice_prog.rt = []
        _practice_prog_allKeys = []
        # store start times for practice
        practice.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice.tStart = globalClock.getTime(format='float')
        practice.status = STARTED
        thisExp.addData('practice.started', practice.tStart)
        practice.maxDuration = None
        # keep track of which components have finished
        practiceComponents = practice.components
        for thisComponent in practice.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice" ---
        # if trial has changed, end Routine now
        if isinstance(practice_trials, data.TrialHandler2) and thisPractice_trial.thisN != practice_trials.thisTrial.thisN:
            continueRoutine = False
        practice.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_2* updates
            
            # if text_2 is starting this frame...
            if text_2.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                text_2.frameNStart = frameN  # exact frame index
                text_2.tStart = t  # local t and not account for scr refresh
                text_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.started')
                # update status
                text_2.status = STARTED
                text_2.setAutoDraw(True)
            
            # if text_2 is active this frame...
            if text_2.status == STARTED:
                # update params
                pass
            
            # *practice_puzzles* updates
            
            # if practice_puzzles is starting this frame...
            if practice_puzzles.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                practice_puzzles.frameNStart = frameN  # exact frame index
                practice_puzzles.tStart = t  # local t and not account for scr refresh
                practice_puzzles.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_puzzles, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_puzzles.started')
                # update status
                practice_puzzles.status = STARTED
                practice_puzzles.setAutoDraw(True)
            
            # if practice_puzzles is active this frame...
            if practice_puzzles.status == STARTED:
                # update params
                pass
            
            # *practice_prog* updates
            waitOnFlip = False
            
            # if practice_prog is starting this frame...
            if practice_prog.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                # keep track of start time/frame for later
                practice_prog.frameNStart = frameN  # exact frame index
                practice_prog.tStart = t  # local t and not account for scr refresh
                practice_prog.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_prog, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_prog.started')
                # update status
                practice_prog.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(practice_prog.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(practice_prog.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if practice_prog.status == STARTED and not waitOnFlip:
                theseKeys = practice_prog.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _practice_prog_allKeys.extend(theseKeys)
                if len(_practice_prog_allKeys):
                    practice_prog.keys = _practice_prog_allKeys[-1].name  # just the last key pressed
                    practice_prog.rt = _practice_prog_allKeys[-1].rt
                    practice_prog.duration = _practice_prog_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                practice.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in practice.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice" ---
        for thisComponent in practice.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice
        practice.tStop = globalClock.getTime(format='float')
        practice.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice.stopped', practice.tStop)
        # check responses
        if practice_prog.keys in ['', [], None]:  # No response was made
            practice_prog.keys = None
        practice_trials.addData('practice_prog.keys',practice_prog.keys)
        if practice_prog.keys != None:  # we had a response
            practice_trials.addData('practice_prog.rt', practice_prog.rt)
            practice_trials.addData('practice_prog.duration', practice_prog.duration)
        # the Routine "practice" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # set up handler to look after randomisation of conditions etc
    block = data.TrialHandler2(
        name='block',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions("DFKI\experiment_design\conditions\ChooseBlocks"+expInfo['group']+".xlsx"), 
        seed=None, 
    )
    thisExp.addLoop(block)  # add the loop to the experiment
    thisBlock = block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
    if thisBlock != None:
        for paramName in thisBlock:
            globals()[paramName] = thisBlock[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlock in block:
        currentLoop = block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlock.rgb)
        if thisBlock != None:
            for paramName in thisBlock:
                globals()[paramName] = thisBlock[paramName]
        
        # --- Prepare to start Routine "session_info" ---
        # create an object to store info about Routine session_info
        session_info = data.Routine(
            name='session_info',
            components=[session_inf, start_session],
        )
        session_info.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        session_inf.setText(readymsg)
        # create starting attributes for start_session
        start_session.keys = []
        start_session.rt = []
        _start_session_allKeys = []
        # store start times for session_info
        session_info.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        session_info.tStart = globalClock.getTime(format='float')
        session_info.status = STARTED
        thisExp.addData('session_info.started', session_info.tStart)
        session_info.maxDuration = None
        # keep track of which components have finished
        session_infoComponents = session_info.components
        for thisComponent in session_info.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "session_info" ---
        # if trial has changed, end Routine now
        if isinstance(block, data.TrialHandler2) and thisBlock.thisN != block.thisTrial.thisN:
            continueRoutine = False
        session_info.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *session_inf* updates
            
            # if session_inf is starting this frame...
            if session_inf.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                session_inf.frameNStart = frameN  # exact frame index
                session_inf.tStart = t  # local t and not account for scr refresh
                session_inf.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(session_inf, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'session_inf.started')
                # update status
                session_inf.status = STARTED
                session_inf.setAutoDraw(True)
            
            # if session_inf is active this frame...
            if session_inf.status == STARTED:
                # update params
                pass
            
            # *start_session* updates
            waitOnFlip = False
            
            # if start_session is starting this frame...
            if start_session.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                start_session.frameNStart = frameN  # exact frame index
                start_session.tStart = t  # local t and not account for scr refresh
                start_session.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(start_session, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'start_session.started')
                # update status
                start_session.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(start_session.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(start_session.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if start_session.status == STARTED and not waitOnFlip:
                theseKeys = start_session.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _start_session_allKeys.extend(theseKeys)
                if len(_start_session_allKeys):
                    start_session.keys = _start_session_allKeys[-1].name  # just the last key pressed
                    start_session.rt = _start_session_allKeys[-1].rt
                    start_session.duration = _start_session_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                session_info.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in session_info.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "session_info" ---
        for thisComponent in session_info.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for session_info
        session_info.tStop = globalClock.getTime(format='float')
        session_info.tStopRefresh = tThisFlipGlobal
        thisExp.addData('session_info.stopped', session_info.tStop)
        # check responses
        if start_session.keys in ['', [], None]:  # No response was made
            start_session.keys = None
        block.addData('start_session.keys',start_session.keys)
        if start_session.keys != None:  # we had a response
            block.addData('start_session.rt', start_session.rt)
            block.addData('start_session.duration', start_session.duration)
        # the Routine "session_info" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        trials = data.TrialHandler2(
            name='trials',
            nReps=1.0, 
            method='sequential', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(conds), 
            seed=None, 
        )
        thisExp.addLoop(trials)  # add the loop to the experiment
        thisTrial = trials.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
        if thisTrial != None:
            for paramName in thisTrial:
                globals()[paramName] = thisTrial[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisTrial in trials:
            currentLoop = trials
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisTrial.rgb)
            if thisTrial != None:
                for paramName in thisTrial:
                    globals()[paramName] = thisTrial[paramName]
            
            # --- Prepare to start Routine "trial" ---
            # create an object to store info about Routine trial
            trial = data.Routine(
                name='trial',
                components=[puzzles, trial_prog, skip_msg],
            )
            trial.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from timer
            if trials.thisN == 0:
                session_timer = core.Clock()
                
            puzzles.setImage(stims)
            # Run 'Begin Routine' code from tag
            import requests
            
            def tag(start_of_trial: bool, puzzle_id: int, condition: str, skip: bool):
                url = "http://localhost:8765/recording/tag"
                data = {
                    "start_of_trial": start_of_trial,
                    "puzzle_id": puzzle_id,
                    "condition": condition,
                    "skip": skip
                }
                request = requests.post(url, json=data)
                print(request.status_code, request.text)
                print(puzzle_id)
            
            #condition = 1 --> low_cl, condition = 2 --> high_cl
            tag(start_of_trial=True, 
                puzzle_id=stims.replace("DFKI\\experiment_design\\stimuli\\", "").replace("high_cl\\tangram", "").replace("low_cl\\tangram", "").replace(".png", "").replace("(", "").replace(")", ""), 
                condition=conds.replace("DFKI\\experiment_design\\conditions\\HighclBlock.xlsx", "2").replace("DFKI\\experiment_design\\conditions\\LowclBlock.xlsx", "1"), 
                skip=False)
            
            # create starting attributes for trial_prog
            trial_prog.keys = []
            trial_prog.rt = []
            _trial_prog_allKeys = []
            # store start times for trial
            trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            trial.tStart = globalClock.getTime(format='float')
            trial.status = STARTED
            thisExp.addData('trial.started', trial.tStart)
            trial.maxDuration = None
            # keep track of which components have finished
            trialComponents = trial.components
            for thisComponent in trial.components:
                thisComponent.tStart = None
                thisComponent.tStop = None
                thisComponent.tStartRefresh = None
                thisComponent.tStopRefresh = None
                if hasattr(thisComponent, 'status'):
                    thisComponent.status = NOT_STARTED
            # reset timers
            t = 0
            _timeToFirstFrame = win.getFutureFlipTime(clock="now")
            frameN = -1
            
            # --- Run Routine "trial" ---
            # if trial has changed, end Routine now
            if isinstance(trials, data.TrialHandler2) and thisTrial.thisN != trials.thisTrial.thisN:
                continueRoutine = False
            trial.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                # Run 'Each Frame' code from timer
                if session_timer.getTime() >= 900:
                    continueRoutine = False
                    trials.finished = True
                
                # *puzzles* updates
                
                # if puzzles is starting this frame...
                if puzzles.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    puzzles.frameNStart = frameN  # exact frame index
                    puzzles.tStart = t  # local t and not account for scr refresh
                    puzzles.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(puzzles, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'puzzles.started')
                    # update status
                    puzzles.status = STARTED
                    puzzles.setAutoDraw(True)
                
                # if puzzles is active this frame...
                if puzzles.status == STARTED:
                    # update params
                    pass
                
                # *trial_prog* updates
                waitOnFlip = False
                
                # if trial_prog is starting this frame...
                if trial_prog.status == NOT_STARTED and tThisFlip >= 0.5-frameTolerance:
                    # keep track of start time/frame for later
                    trial_prog.frameNStart = frameN  # exact frame index
                    trial_prog.tStart = t  # local t and not account for scr refresh
                    trial_prog.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(trial_prog, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'trial_prog.started')
                    # update status
                    trial_prog.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(trial_prog.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(trial_prog.clearEvents, eventType='keyboard')  # clear events on next screen flip
                if trial_prog.status == STARTED and not waitOnFlip:
                    theseKeys = trial_prog.getKeys(keyList=['space','s'], ignoreKeys=["escape"], waitRelease=False)
                    _trial_prog_allKeys.extend(theseKeys)
                    if len(_trial_prog_allKeys):
                        trial_prog.keys = _trial_prog_allKeys[-1].name  # just the last key pressed
                        trial_prog.rt = _trial_prog_allKeys[-1].rt
                        trial_prog.duration = _trial_prog_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *skip_msg* updates
                
                # if skip_msg is starting this frame...
                if skip_msg.status == NOT_STARTED and tThisFlip >= 300-frameTolerance:
                    # keep track of start time/frame for later
                    skip_msg.frameNStart = frameN  # exact frame index
                    skip_msg.tStart = t  # local t and not account for scr refresh
                    skip_msg.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(skip_msg, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'skip_msg.started')
                    # update status
                    skip_msg.status = STARTED
                    skip_msg.setAutoDraw(True)
                
                # if skip_msg is active this frame...
                if skip_msg.status == STARTED:
                    # update params
                    pass
                
                # check for quit (typically the Esc key)
                if defaultKeyboard.getKeys(keyList=["escape"]):
                    thisExp.status = FINISHED
                if thisExp.status == FINISHED or endExpNow:
                    endExperiment(thisExp, win=win)
                    return
                # pause experiment here if requested
                if thisExp.status == PAUSED:
                    pauseExperiment(
                        thisExp=thisExp, 
                        win=win, 
                        timers=[routineTimer], 
                        playbackComponents=[]
                    )
                    # skip the frame we paused on
                    continue
                
                # check if all components have finished
                if not continueRoutine:  # a component has requested a forced-end of Routine
                    trial.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in trial.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "trial" ---
            for thisComponent in trial.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for trial
            trial.tStop = globalClock.getTime(format='float')
            trial.tStopRefresh = tThisFlipGlobal
            thisExp.addData('trial.stopped', trial.tStop)
            # Run 'End Routine' code from tag
            #condition = 1 --> low_cl, condition = 2 --> high_cl
            
            #end of the trial post request
            if trial_prog.keys == 's':
                tag(start_of_trial=False, 
                    puzzle_id=stims.replace("DFKI\\experiment_design\\stimuli\\", "").replace("high_cl\\tangram", "").replace("low_cl\\tangram", "").replace(".png", "").replace("(", "").replace(")", ""), 
                    condition=conds.replace("DFKI\\experiment_design\\conditions\\HighclBlock.xlsx", "2").replace("DFKI\\experiment_design\\conditions\\LowclBlock.xlsx", "1"), 
                    skip=True)
            elif trial_prog.keys == 'space':
                tag(start_of_trial=False, 
                    puzzle_id=stims.replace("DFKI\\experiment_design\\stimuli\\", "").replace("high_cl\\tangram", "").replace("low_cl\\tangram", "").replace(".png", "").replace("(", "").replace(")", ""), 
                    condition=conds.replace("DFKI\\experiment_design\\conditions\\HighclBlock.xlsx", "2").replace("DFKI\\experiment_design\\conditions\\LowclBlock.xlsx", "1"), 
                    skip=False)
            # check responses
            if trial_prog.keys in ['', [], None]:  # No response was made
                trial_prog.keys = None
            trials.addData('trial_prog.keys',trial_prog.keys)
            if trial_prog.keys != None:  # we had a response
                trials.addData('trial_prog.rt', trial_prog.rt)
                trials.addData('trial_prog.duration', trial_prog.duration)
            # the Routine "trial" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'trials'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'block'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "bye" ---
    # create an object to store info about Routine bye
    bye = data.Routine(
        name='bye',
        components=[bye_txt],
    )
    bye.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # store start times for bye
    bye.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    bye.tStart = globalClock.getTime(format='float')
    bye.status = STARTED
    thisExp.addData('bye.started', bye.tStart)
    bye.maxDuration = None
    # keep track of which components have finished
    byeComponents = bye.components
    for thisComponent in bye.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "bye" ---
    bye.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine and routineTimer.getTime() < 10.0:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *bye_txt* updates
        
        # if bye_txt is starting this frame...
        if bye_txt.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            bye_txt.frameNStart = frameN  # exact frame index
            bye_txt.tStart = t  # local t and not account for scr refresh
            bye_txt.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(bye_txt, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'bye_txt.started')
            # update status
            bye_txt.status = STARTED
            bye_txt.setAutoDraw(True)
        
        # if bye_txt is active this frame...
        if bye_txt.status == STARTED:
            # update params
            pass
        
        # if bye_txt is stopping this frame...
        if bye_txt.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > bye_txt.tStartRefresh + 10.0-frameTolerance:
                # keep track of stop time/frame for later
                bye_txt.tStop = t  # not accounting for scr refresh
                bye_txt.tStopRefresh = tThisFlipGlobal  # on global time
                bye_txt.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'bye_txt.stopped')
                # update status
                bye_txt.status = FINISHED
                bye_txt.setAutoDraw(False)
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            bye.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in bye.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "bye" ---
    for thisComponent in bye.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for bye
    bye.tStop = globalClock.getTime(format='float')
    bye.tStopRefresh = tThisFlipGlobal
    thisExp.addData('bye.stopped', bye.tStop)
    # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
    if bye.maxDurationReached:
        routineTimer.addTime(-bye.maxDuration)
    elif bye.forceEnded:
        routineTimer.reset()
    else:
        routineTimer.addTime(-10.000000)
    thisExp.nextEntry()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='comma')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
