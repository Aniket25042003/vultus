import { useEffect, useRef, useState, type FormEvent } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import api from '../lib/api';
import { clearSession, getUser, updateStoredUser } from '../lib/auth';
import { Badge } from '../components/ui/badge';
import { Button } from '../components/ui/button';
import { Card, CardDescription, CardTitle } from '../components/ui/card';
import { Progress } from '../components/ui/progress';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { Input } from '../components/ui/input';
import { Logo } from '../components/ui/logo';
const DashboardPage = () => {
  const navigate = useNavigate();
  const [user, setUser] = useState(() => getUser());
  const [tab, setTab] = useState('teacher');
  const [teacherData, setTeacherData] = useState<any>(null);
  const [adminData, setAdminData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [sampleResponse, setSampleResponse] = useState<any>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [lastScan, setLastScan] = useState<any>(null);
  const [pendingOverride, setPendingOverride] = useState<any>(null);
  const [scanBusy, setScanBusy] = useState(false);
  const [overrideBusy, setOverrideBusy] = useState(false);
  const [enrollName, setEnrollName] = useState('');
  const [enrollStudentId, setEnrollStudentId] = useState('');
  const [enrollFiles, setEnrollFiles] = useState<File[]>([]);
  const [enrollBusy, setEnrollBusy] = useState(false);
  const [enrollResult, setEnrollResult] = useState<any>(null);
  const [students, setStudents] = useState<any[]>([]);
  const [studentsLoading, setStudentsLoading] = useState(false);
  const [studentsError, setStudentsError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const inFlightRef = useRef(false);

  const refreshDashboards = async (currentUser?: any) => {
    const activeUser = currentUser ?? user;
    if (!activeUser) return;
    const teacherResponse = await api.get('/api/dashboard/teacher');
    setTeacherData(teacherResponse.data);
    if (activeUser.role === 'admin') {
      const adminResponse = await api.get('/api/dashboard/admin');
      setAdminData(adminResponse.data);
      setStudentsLoading(true);
      setStudentsError(null);
      try {
        const studentResponse = await api.get('/api/students');
        setStudents(studentResponse.data.students ?? []);
      } catch (error: any) {
        setStudentsError(error?.response?.data?.error ?? 'Failed to load students.');
      } finally {
        setStudentsLoading(false);
      }
    }
  };

  useEffect(() => {
    const load = async () => {
      try {
        const meResponse = await api.get('/api/auth/me');
        const me = meResponse.data.user;
        const validatedUser = {
          id: me.id,
          email: me.email,
          role: me.role,
          fullName: me.name ?? me.fullName,
        };
        updateStoredUser(validatedUser);
        setUser(validatedUser);
        setTab(me.role === 'admin' ? 'admin' : 'teacher');
        await refreshDashboards(validatedUser);
      } catch (error) {
        clearSession();
        navigate('/login');
      } finally {
        setLoading(false);
      }
    };

    load();
  }, [navigate]);

  useEffect(() => {
    if (!user) {
      setTab('teacher');
      return;
    }
    if (user.role !== 'admin' && tab === 'admin') {
      setTab('teacher');
    }
  }, [tab, user]);

  useEffect(() => {
    if (!user) return;
    const interval = setInterval(() => {
      refreshDashboards().catch(() => null);
    }, 8000);
    return () => clearInterval(interval);
  }, [user]);

  useEffect(() => {
    if (!cameraActive) {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      return;
    }
    const start = async () => {
      setCameraError(null);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        });
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
      } catch (error: any) {
        setCameraError(error?.message ?? 'Unable to access camera.');
        setCameraActive(false);
      }
    };
    start();
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, [cameraActive]);

  const drawBoundingBox = (faceBox: number[] | null, decision: string | null) => {
    const overlay = overlayRef.current;
    const video = videoRef.current;
    if (!overlay || !video) return;
    overlay.width = video.videoWidth;
    overlay.height = video.videoHeight;
    const ctx = overlay.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, overlay.width, overlay.height);
    if (!faceBox || faceBox.length < 4) return;
    const [x1, y1, x2, y2] = faceBox;
    const color = decision === 'granted' ? '#22c55e' : decision === 'denied' ? '#ef4444' : '#facc15';
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.shadowBlur = 0;
    if (decision) {
      ctx.fillStyle = color;
      ctx.font = 'bold 14px Inter, system-ui, sans-serif';
      const label = decision.toUpperCase();
      const textWidth = ctx.measureText(label).width;
      ctx.fillRect(x1, y1 - 22, textWidth + 12, 22);
      ctx.fillStyle = '#fff';
      ctx.fillText(label, x1 + 6, y1 - 6);
    }
  };

  useEffect(() => {
    if (!cameraActive) {
      // Clear overlay when camera stops
      const overlay = overlayRef.current;
      if (overlay) {
        const ctx = overlay.getContext('2d');
        ctx?.clearRect(0, 0, overlay.width, overlay.height);
      }
      return;
    }
    const interval = setInterval(async () => {
      if (inFlightRef.current || !videoRef.current || !canvasRef.current) return;
      if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) return;
      inFlightRef.current = true;
      setScanBusy(true);
      try {
        const canvas = canvasRef.current;
        const context = canvas.getContext('2d');
        if (!context) return;
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
        const dataUrl = canvas.toDataURL('image/jpeg', 0.8);
        const imageBase64 = dataUrl.split(',')[1];

        // Step 1: Detect face using YOLOv8-Face (fast)
        const personRes = await api.post('/api/detections/person', {
          imageBase64,
          source: 'admin-camera',
        });

        if (!personRes.data?.detected) {
          // No face detected — clear bounding box, skip expensive embedding pipeline
          drawBoundingBox(null, null);
          setLastScan((prev: any) => prev?.decision ? prev : { decision: null, personDetected: false });
          return;
        }

        // Step 2: Face detected — run full pipeline (crop + IResNet embedding + DB matching)
        const response = await api.post('/api/detections/face', {
          imageBase64,
          source: 'admin-camera',
        });
        const scan = response.data;
        drawBoundingBox(scan.faceBox ?? null, scan.decision ?? null);
        setLastScan(scan);

        // Preserve any denied scan for the override button
        if (scan.decision === 'denied' && scan.accessEventId && scan.personDetected) {
          setPendingOverride(scan);
        }

        await refreshDashboards();
      } catch (error: any) {
        setLastScan({ error: error?.response?.data?.error ?? 'Scan failed.' });
      } finally {
        inFlightRef.current = false;
        setScanBusy(false);
      }
    }, 1400);
    return () => clearInterval(interval);
  }, [cameraActive]);

  const runSample = async (endpoint: string) => {
    try {
      const response = await api.post(endpoint, { frameId: `demo-${Date.now()}` });
      setSampleResponse(response.data);
    } catch (error: any) {
      setSampleResponse({ error: error?.response?.data?.error ?? 'Request failed.' });
    }
  };

  const handleOverride = async () => {
    const target = pendingOverride;
    if (!target?.accessEventId) return;
    setOverrideBusy(true);
    try {
      const response = await api.post('/api/access/override', {
        accessEventId: target.accessEventId,
        decision: 'granted',
        reason: 'manual override',
      });
      setPendingOverride(null);
      setLastScan((prev: any) => {
        if (prev?.accessEventId === target.accessEventId) {
          return {
            ...prev,
            decision: 'granted',
            overridden: true,
            override_at: response.data.override_at ?? new Date().toISOString(),
          };
        }
        return prev;
      });
      await refreshDashboards();
    } catch (error: any) {
      setLastScan((prev: any) => ({
        ...prev,
        error: error?.response?.data?.error ?? 'Override failed.',
      }));
    } finally {
      setOverrideBusy(false);
    }
  };

  const fileToBase64 = (file: File) =>
    new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => {
        const result = typeof reader.result === 'string' ? reader.result : '';
        const base64 = result.includes(',') ? result.split(',')[1] : result;
        resolve(base64);
      };
      reader.onerror = () => reject(new Error('Failed to read image.'));
      reader.readAsDataURL(file);
    });

  const handleEnroll = async (event: FormEvent) => {
    event.preventDefault();
    if (!enrollName.trim() || !enrollStudentId.trim() || enrollFiles.length === 0) {
      setEnrollResult({ error: 'Student name, Student ID, and at least one image are required.' });
      return;
    }
    setEnrollBusy(true);
    setEnrollResult(null);
    try {
      const images = await Promise.all(enrollFiles.map((file) => fileToBase64(file)));
      const response = await api.post('/api/students/enroll', {
        fullName: enrollName.trim(),
        label: enrollStudentId.trim(),
        images,
      });
      const data = response.data;
      if (data.stored > 0) {
        setEnrollResult({
          success: true,
          message: `Successfully enrolled "${data.fullName}" (ID: ${data.externalLabel}) with ${data.stored} image${data.stored > 1 ? 's' : ''}.${data.failed > 0 ? ` ${data.failed} image(s) failed to process.` : ''}`,
        });
      } else {
        setEnrollResult({ error: `Enrollment failed — none of the ${data.failed} image(s) could be processed. Please try with clearer face photos.` });
      }
      setEnrollName('');
      setEnrollStudentId('');
      setEnrollFiles([]);
      await refreshDashboards();
    } catch (error: any) {
      setEnrollResult({ error: error?.response?.data?.error ?? 'Enrollment failed.' });
    } finally {
      setEnrollBusy(false);
    }
  };

  const handleDeleteStudent = async (studentId: string) => {
    const confirmed = window.confirm('Delete this student and their embeddings?');
    if (!confirmed) return;
    setStudentsLoading(true);
    setStudentsError(null);
    try {
      await api.delete(`/api/students/${studentId}`);
      await refreshDashboards();
    } catch (error: any) {
      setStudentsError(error?.response?.data?.error ?? 'Failed to delete student.');
    } finally {
      setStudentsLoading(false);
    }
  };

  if (!user) {
    return null;
  }

  if (loading) {
    return <div className="p-10">Loading dashboard...</div>;
  }

  const accessGranted = teacherData?.accessSummary?.find((row: any) => row.decision === 'granted');
  const accessDenied = teacherData?.accessSummary?.find((row: any) => row.decision === 'denied');
  const totalAccess = (accessGranted?.count ?? 0) + (accessDenied?.count ?? 0);
  const grantRate = totalAccess ? Math.round(((accessGranted?.count ?? 0) / totalAccess) * 100) : 0;

  const rangeItems = [
    { key: 'day', label: '24h' },
    { key: 'week', label: '7d' },
    { key: 'month', label: '30d' },
    { key: 'year', label: '365d' },
  ];

  return (
    <div className="min-h-screen px-6 py-10">
      <header className="mx-auto flex w-full max-w-6xl items-center justify-between">
        <div className="flex items-center gap-6">
          <Logo className="h-12 w-12" />

          <div>
            <p className="text-sm uppercase tracking-[0.25em] text-black/40">VULTUS</p>
            <h1 className="text-3xl font-semibold">
              Welcome, {user.fullName ?? user.email}
            </h1>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <Badge tone="neutral">{user.role}</Badge>
          <Button
            variant="outline"
            onClick={() => {
              clearSession();
              navigate('/login');
            }}
          >
            Log out
          </Button>
        </div>
      </header>

      <main className="mx-auto mt-10 flex w-full max-w-6xl flex-col gap-8">
        {user.role === 'admin' ? (
          <Tabs value={tab} onValueChange={setTab}>
            <TabsList>
              <TabsTrigger value="teacher">Teacher View</TabsTrigger>
              <TabsTrigger value="admin">Admin View</TabsTrigger>
            </TabsList>
            <TabsContent value="teacher">{renderTeacher()}</TabsContent>
            <TabsContent value="admin">{renderAdmin()}</TabsContent>
          </Tabs>
        ) : (
          renderTeacher()
        )}

        {user.role === 'admin' && renderCamera()}
      </main>
    </div>
  );

  function renderTeacher() {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <Card>
            <CardTitle>Emotion Snapshot</CardTitle>
            <CardDescription>Last 24 hours across active classrooms.</CardDescription>
            <div className="mt-6 space-y-4">
              {(teacherData?.emotionSummary ?? []).map((row: any) => (
                <div key={row.emotion} className="space-y-2">
                  <div className="flex items-center justify-between text-sm">
                    <span className="font-medium capitalize">{row.emotion}</span>
                    <span className="text-black/60">{row.count} signals</span>
                  </div>
                  <Progress value={Math.min(100, row.count * 12)} />
                </div>
              ))}
              {teacherData?.emotionSummary?.length === 0 && (
                <p className="text-sm text-black/60">No emotion data yet.</p>
              )}
            </div>
          </Card>
          <Card>
            <CardTitle>Access Integrity</CardTitle>
            <CardDescription>Entry outcomes over the last 24 hours.</CardDescription>
            <div className="mt-6 space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-sm text-black/60">Grant rate</span>
                <span className="text-2xl font-semibold">{grantRate}%</span>
              </div>
              <Progress value={grantRate} />
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="rounded-2xl bg-emerald-500/10 px-4 py-3">
                  <p className="text-black/60">Granted</p>
                  <p className="text-lg font-semibold">{accessGranted?.count ?? 0}</p>
                </div>
                <div className="rounded-2xl bg-rose-500/10 px-4 py-3">
                  <p className="text-black/60">Denied</p>
                  <p className="text-lg font-semibold">{accessDenied?.count ?? 0}</p>
                </div>
              </div>
            </div>
          </Card>
        </div>

        <Card className="mt-6">
          <CardTitle>Attendance Over Time</CardTitle>
          <CardDescription>Access outcomes for recent ranges.</CardDescription>
          <div className="mt-6 grid gap-4 md:grid-cols-4">
            {rangeItems.map((range) => {
              const access = teacherData?.accessRanges?.[range.key];
              const emotion = teacherData?.emotionRanges?.[range.key];
              return (
                <div key={range.key} className="rounded-2xl bg-white/70 px-4 py-3">
                  <p className="text-xs uppercase text-black/50">{range.label}</p>
                  <p className="text-lg font-semibold">
                    {access?.granted ?? 0} granted · {access?.denied ?? 0} denied
                  </p>
                  <p className="text-xs text-black/50">
                    {emotion?.total ?? 0} emotion signals
                    {emotion?.topEmotion ? ` · top ${emotion.topEmotion}` : ''}
                  </p>
                </div>
              );
            })}
          </div>
        </Card>

        <Card className="mt-6">
          <CardTitle>Recent Access Events</CardTitle>
          <CardDescription>Latest decisions made by the face model — with face snapshots for review.</CardDescription>
          <div className="mt-6 space-y-3">
            {(teacherData?.recentAccess ?? []).map((event: any, index: number) => (
              <div
                key={event.id ?? `${event.person_label ?? 'unknown'}-${index}`}
                className="flex items-center gap-4 rounded-2xl bg-white/70 px-4 py-3"
              >
                {event.face_snapshot ? (
                  <img
                    src={`data:image/jpeg;base64,${event.face_snapshot}`}
                    alt="Face"
                    className="h-14 w-14 flex-shrink-0 rounded-xl object-cover ring-2 ring-black/10"
                  />
                ) : (
                  <div className="flex h-14 w-14 flex-shrink-0 items-center justify-center rounded-xl bg-black/5 text-xs text-black/30">
                    No face
                  </div>
                )}
                <div className="min-w-0 flex-1">
                  <p className="font-medium">{event.display_name ?? event.person_label ?? 'Unknown visitor'}</p>
                  {event.external_label && (
                    <p className="text-xs text-black/40">ID: {event.external_label}</p>
                  )}
                  <p className="text-xs text-black/50">
                    {new Date(event.created_at).toLocaleString()}
                    {event.source ? ` · ${event.source}` : ''}
                    {event.match_confidence ? ` · Match: ${(Number(event.match_confidence) * 100).toFixed(1)}%` : ''}
                    {event.overridden && event.override_at
                      ? ` · Overridden ${new Date(event.override_at).toLocaleString()}`
                      : ''}
                  </p>
                </div>
                <Badge tone={event.decision === 'granted' ? 'success' : 'warning'}>
                  {event.decision}
                </Badge>
              </div>
            ))}
            {teacherData?.recentAccess?.length === 0 && (
              <p className="text-sm text-black/60">No access events logged yet.</p>
            )}
          </div>
        </Card>
      </motion.div>
    );
  }

  function renderAdmin() {
    return (
      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="grid gap-6 lg:grid-cols-3">
          <Card>
            <CardTitle>Access Events</CardTitle>
            <p className="mt-4 text-3xl font-semibold">{adminData?.totals?.accessEvents ?? 0}</p>
            <p className="text-sm text-black/60">Total lifetime decisions.</p>
          </Card>
          <Card>
            <CardTitle>Emotion Signals</CardTitle>
            <p className="mt-4 text-3xl font-semibold">{adminData?.totals?.emotionEvents ?? 0}</p>
            <p className="text-sm text-black/60">Insights captured this term.</p>
          </Card>
          <Card>
            <CardTitle>System Readiness</CardTitle>
            <p className="mt-4 text-3xl font-semibold">99.97%</p>
            <p className="text-sm text-black/60">Rolling uptime estimate.</p>
          </Card>
        </div>

        <Card className="mt-6">
          <CardTitle>Enroll Student</CardTitle>
          <CardDescription>Upload one or more images to register a student.</CardDescription>
          <form className="mt-4 space-y-4" onSubmit={handleEnroll}>
            <div className="grid gap-3 md:grid-cols-2">
              <Input
                placeholder="Student full name"
                value={enrollName}
                onChange={(event) => setEnrollName(event.target.value)}
                required
              />
              <Input
                placeholder="Student ID (e.g. 1562262)"
                value={enrollStudentId}
                onChange={(event) => setEnrollStudentId(event.target.value)}
                required
              />
            </div>
            <Input
              type="file"
              accept="image/png,image/jpeg"
              multiple
              onChange={(event) => setEnrollFiles(Array.from(event.target.files ?? []))}
            />
            <div className="flex flex-wrap items-center gap-3">
              <Button type="submit" disabled={enrollBusy}>
                {enrollBusy ? 'Enrolling…' : 'Enroll Student'}
              </Button>
              <Badge tone="neutral">{enrollFiles.length} images selected</Badge>
            </div>
            {enrollResult && (
              <div
                className={`rounded-2xl px-4 py-3 text-sm font-medium ${enrollResult.success
                  ? 'bg-emerald-500/10 text-emerald-700'
                  : 'bg-rose-500/10 text-rose-700'
                  }`}
              >
                {enrollResult.success ? '✓ ' : '✗ '}
                {enrollResult.message ?? enrollResult.error}
              </div>
            )}
          </form>
        </Card>

        <Card className="mt-6">
          <CardTitle>Manage Students</CardTitle>
          <CardDescription>Review enrollments, usage, and clean up records.</CardDescription>
          <div className="mt-4 space-y-3">
            {studentsLoading && <p className="text-sm text-black/60">Loading students…</p>}
            {students.map((student) => (
              <div
                key={student.id}
                className="flex flex-col gap-3 rounded-2xl bg-white/70 px-4 py-3 md:flex-row md:items-center md:justify-between"
              >
                <div>
                  <p className="font-medium">{student.full_name}</p>
                  <p className="text-xs text-black/50">{student.external_label}</p>
                  <div className="mt-2 flex flex-wrap gap-2 text-xs text-black/60">
                    <span>{student.embedding_count} embeddings</span>
                    <span>{student.access_granted} granted</span>
                    <span>{student.access_denied} denied</span>
                    <span>{student.emotion_total} emotions</span>
                  </div>
                  <p className="mt-1 text-xs text-black/50">
                    Last access:{' '}
                    {student.last_access_at ? new Date(student.last_access_at).toLocaleString() : '—'}
                  </p>
                </div>
                <Button
                  variant="outline"
                  onClick={() => handleDeleteStudent(student.id)}
                  disabled={studentsLoading}
                >
                  Delete
                </Button>
              </div>
            ))}
            {!studentsLoading && !students.length && (
              <p className="text-sm text-black/60">No students enrolled yet.</p>
            )}
            {studentsError && <p className="text-sm text-rose-500">{studentsError}</p>}
          </div>
        </Card>


        <Card className="mt-6">
          <CardTitle>Decision Rates (7 days)</CardTitle>
          <CardDescription>Grant vs deny distribution across all gates.</CardDescription>
          <div className="mt-6 grid gap-4 md:grid-cols-2">
            {(adminData?.decisionRates ?? []).map((row: any) => (
              <div key={row.decision} className="rounded-2xl bg-white/70 px-4 py-3">
                <p className="text-sm uppercase text-black/50">{row.decision}</p>
                <p className="text-2xl font-semibold">{row.count}</p>
              </div>
            ))}
            {adminData?.decisionRates?.length === 0 && (
              <p className="text-sm text-black/60">No decisions yet.</p>
            )}
          </div>
        </Card>

        <Card className="mt-6">
          <CardTitle>Model Sandbox</CardTitle>
          <CardDescription>Quick sanity checks for the live inference endpoints.</CardDescription>
          <div className="mt-4 flex flex-wrap gap-3">
            <Button variant="outline" onClick={() => runSample('/api/detections/person')}>
              Run Person Detection
            </Button>
            <Button variant="outline" onClick={() => runSample('/api/detections/face')}>
              Run Access Pipeline
            </Button>
            <Button variant="outline" onClick={() => runSample('/api/detections/emotion')}>
              Run Emotion Detection
            </Button>
          </div>
          {sampleResponse && (
            <pre className="mt-4 rounded-2xl bg-black/90 p-4 text-xs text-emerald-200">
              {JSON.stringify(sampleResponse, null, 2)}
            </pre>
          )}
        </Card>
      </motion.div>
    );
  }

  function renderCamera() {
    return (
      <Card className="mt-6">
        <CardTitle>Live Camera Feed</CardTitle>
        <CardDescription>Monitor entries and manually override if needed.</CardDescription>
        <div className="mt-4 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="space-y-4">
            <div className="relative overflow-hidden rounded-2xl bg-black/80">
              <video ref={videoRef} className="aspect-video w-full object-cover" muted playsInline />
              <canvas
                ref={overlayRef}
                className="pointer-events-none absolute inset-0 h-full w-full"
                style={{ objectFit: 'cover' }}
              />
              {!cameraActive && (
                <div className="absolute inset-0 flex items-center justify-center text-sm text-white/70">
                  Camera is off
                </div>
              )}
            </div>
            <div className="flex flex-wrap gap-3">
              <Button variant={cameraActive ? 'outline' : 'primary'} onClick={() => setCameraActive(!cameraActive)}>
                {cameraActive ? 'Stop Camera' : 'Start Camera'}
              </Button>
              <Badge tone={scanBusy ? 'warning' : 'neutral'}>
                {scanBusy ? 'Scanning…' : 'Idle'}
              </Badge>
            </div>
            {cameraError && <p className="text-sm text-rose-500">{cameraError}</p>}
          </div>
          <div className="space-y-4">
            <div className="rounded-2xl bg-white/70 px-4 py-4">
              <p className="text-xs uppercase text-black/50">Last scan</p>
              <p className="mt-2 text-lg font-semibold">
                {lastScan?.personLabel ?? lastScan?.person_label ?? 'Unknown'}
              </p>
              <p className="text-sm text-black/60">
                {lastScan?.decision ? `Decision: ${lastScan.decision}` : 'No scans yet'}
              </p>
              {lastScan?.matchConfidence != null && lastScan.matchConfidence > 0 && (
                <p className="text-sm text-black/60">
                  Match confidence: {(Number(lastScan.matchConfidence) * 100).toFixed(1)}%
                </p>
              )}
              {lastScan?.emotion && (
                <p className="text-sm text-black/60">
                  Emotion: {lastScan.emotion.emotion} ({lastScan.emotion.confidence})
                </p>
              )}
              {lastScan?.timestamp && (
                <p className="text-xs text-black/50">
                  {new Date(lastScan.timestamp).toLocaleString()}
                </p>
              )}
            </div>
            {pendingOverride && (
              <div className="rounded-2xl border border-amber-300/50 bg-amber-50/80 px-4 py-3 space-y-2">
                <p className="text-xs font-semibold text-amber-800">⚠ Denied Access Pending Review</p>
                <p className="text-xs text-amber-700">
                  {pendingOverride.personLabel ?? 'Unknown'} was denied at{' '}
                  {pendingOverride.timestamp ? new Date(pendingOverride.timestamp).toLocaleTimeString() : '—'}
                </p>
                <div className="flex gap-2">
                  <Button
                    variant="primary"
                    disabled={overrideBusy}
                    onClick={handleOverride}
                  >
                    {overrideBusy ? 'Granting…' : '✓ Grant Access'}
                  </Button>
                  <Button variant="outline" onClick={() => setPendingOverride(null)}>
                    Dismiss
                  </Button>
                </div>
              </div>
            )}
            {lastScan?.error && <p className="text-sm text-rose-500">{lastScan.error}</p>}
          </div>
        </div>
        <canvas ref={canvasRef} className="hidden" />
      </Card>
    );
  }
};

export default DashboardPage;
