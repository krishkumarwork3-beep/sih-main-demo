# URGENT: Camera 4 (cam4) Investigation Required

## PROBLEM STATEMENT

Camera 4 is showing status "offline" but there is evidence it **was** running at some point. Need to investigate why it's not running/playing properly now.

## EVIDENCE GATHERED

### ✅ Video File Exists
- **Path**: `d:\demo\ibvap_demo\video4.mp4`
- **File size**: 1,420,663 bytes (~1.4 MB)
- **Last modified**: September 24, 2026, 9:18:59 PM
- **File is readable**: Yes (PowerShell confirmed)

### ❌ Current Status: OFFLINE
From `web/health_cam4.json`:
```json
{
  "camera_id": "cam4",
  "status": "offline",
  "fps": 0.0,
  "blank_frame": false,
  "last_update": 1790289795.7607374
}
```

### ⚠️ Evidence It WAS Running Previously
From `web/detections_cam4.json` - shows 9 face detection entries with multiple track IDs:
- Track IDs detected: 1, 13, 14, 24, 25, 33
- Timestamps: Between 04:10:56 and 04:13:13
- All labeled as "no_usable_face" (expected for a parked vehicle video)

**This proves cam4 DID run successfully at some point and detected objects (wrongly classified as persons with faces, but that's a separate issue).**

### 🖼️ Latest Frame Exists
- File `web/latest_frame_cam4.jpg` exists
- This is the last frame cam4 captured before going offline

---

## INVESTIGATION REQUIRED

### 1. **Why Did cam4 Go Offline?**

Check the following in `pipeline.py`:

**A. Video End Behavior**
- Is cam4's video shorter than the other videos (cam1, cam2, cam3)?
- When cam4's video ends, what happens?
- Is `--loop` flag being used when running `camera_manager.py`?
- If NO loop: cam4 should show status "stream_ended" not "offline"
- If YES loop: cam4 should restart the video automatically

**B. Potential Crash/Error**
- Did cam4's process crash during execution?
- Check if there are any Python exceptions in the console output
- Check if OpenCV failed to read frames from video4.mp4
- Check if the FFMPEG backend had codec issues with this specific video

**C. Video Duration Mismatch**
```python
# In pipeline.py, check this for cam4:
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS) or 25
print(f"[cam4] opened video4.mp4: ~{total_frames / fps:.1f}s at {fps:.1f} fps ({int(total_frames)} frames).")
```

**Action**: Run the demo and capture the console output showing cam4's video properties. Compare with cam1, cam2, cam3.

Expected output for all cameras should show:
- cam1: `[cam1] opened video.mp4: ~XXs at YY fps`
- cam2: `[cam2] opened video2.mp4: ~XXs at YY fps`
- cam3: `[cam3] opened video3.mp4: ~XXs at YY fps`
- cam4: `[cam4] opened video4.mp4: ~XXs at YY fps`

If cam4's duration is much shorter, that's the issue.

---

### 2. **False Person Detections in cam4**

**Observation**: cam4's detections log shows multiple track IDs (1, 13, 14, 24, 25, 33) with "no_usable_face" labels.

**Expected behavior**: video4.mp4 is supposed to be a **parked vehicle at night** with **NO person** in it.

**Problem**: YOLO is detecting something as class 0 (person) when it should only detect class 2/3/5/7 (vehicles).

**Possible causes**:
1. The parked vehicle's shape/silhouette is being misclassified as a person by YOLO
2. There ARE people in video4.mp4 that we didn't know about
3. Shadows/reflections/objects near the vehicle are triggering false person detections

**Action needed**:
- Manually review video4.mp4 to confirm there are NO people in the frame
- If YOLO is falsely detecting the vehicle as a person, this is a known YOLO limitation (not a bug in the code)
- Consider adjusting YOLO confidence threshold for cam4 specifically if false positives are excessive

---

### 3. **Video Synchronization Impact**

**Current behavior**: cam4 goes offline while cam1, cam2, cam3 presumably keep running.

**For the demo to work properly**:
- All 4 cameras should start at the same time
- All 4 cameras should play at the same pace
- If one video is shorter, either:
  - Loop it immediately (if `--loop` flag is set)
  - Pause/freeze on last frame while others continue
  - Stop the entire demo when the shortest video ends

**Currently**: cam4 is going offline which suggests it's ending before the others, but NOT looping properly.

---

## REQUESTED FIXES

### Fix 1: Verify Video Properties
Add debug output to show duration/frame count for all 4 videos at startup:

```python
# In pipeline.py, after opening the video:
print(f"[{camera_id}] Video properties:")
print(f"  - Total frames: {int(total_frames)}")
print(f"  - FPS: {fps:.1f}")
print(f"  - Duration: {total_frames / fps:.2f} seconds")
print(f"  - Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
```

### Fix 2: Ensure Looping Works for cam4
Check the looping logic in `pipeline.py`:

```python
ok, frame = cap.read()
if not ok:
    if args.loop and source_path not in ("0", 0):
        write_health(camera_id, "stream_ended_looping", 0.0, False)
        cap.release()
        cap = open_capture()
        continue  # ← Does this work for cam4?
    write_health(camera_id, "stream_ended", 0.0, False)
    print(f"[{camera_id}] end of stream.")
    break
```

**Potential issues**:
- Does `open_capture()` successfully reopen video4.mp4?
- Is there a race condition where cam4 marks itself offline before looping?
- Does the health status get updated correctly when looping?

### Fix 3: Make cam4 Loop Indefinitely (If Shorter Than Others)
If video4.mp4 is significantly shorter than video.mp4, video2.mp4, video3.mp4:

**Option A**: Always loop cam4, even if `--loop` is not set
```python
# Special handling for cam4 since it's just a static vehicle scene
if camera_id == "cam4" and not ok:
    cap.release()
    cap = open_capture()
    continue
```

**Option B**: Freeze cam4 on last frame when it ends
```python
if not ok and last_frame is not None:
    frame = last_frame.copy()  # Keep showing the last frame
    ok = True
```

---

## TESTING STEPS

### Step 1: Check Current Video Durations
```bash
# Use ffprobe or similar to check video lengths
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video.mp4
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video2.mp4
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video3.mp4
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 video4.mp4
```

### Step 2: Run Demo with Loop Enabled
```bash
python camera_manager.py --device cuda --loop
```

**Watch for**:
- Does cam4 print "opened video4.mp4" with duration?
- Does cam4 run for a while then go offline?
- Does cam4 restart when it reaches the end?
- Are there any errors in console for cam4 specifically?

### Step 3: Check Latest Frame
Open `web/latest_frame_cam4.jpg` to see what cam4 was showing when it went offline:
- Is it a valid frame from video4.mp4?
- Is it the last frame of the video (suggesting it reached the end)?
- Is it corrupted/blank (suggesting a read error)?

---

## EXPECTED OUTCOMES AFTER FIX

1. ✅ cam4 runs continuously alongside cam1, cam2, cam3
2. ✅ cam4 shows status "healthy" or "stream_ended_looping" (not "offline")
3. ✅ cam4's live feed updates on the dashboard in real-time
4. ✅ If cam4 is shorter, it loops seamlessly without going offline
5. ✅ Console output shows cam4's video properties (duration, fps, resolution)
6. ✅ No errors/exceptions for cam4 in console output

---

## PRIORITY: HIGH

This is blocking the multi-camera demo from working properly. All 4 cameras must be running simultaneously for the cross-camera tracking demonstration to work.

**Next steps**: 
1. Run the demo and capture full console output
2. Check video4.mp4 duration vs other videos
3. Verify looping behavior
4. Fix the offline issue
5. Test with all 4 cameras running continuously
