import { useState, useEffect } from 'react';
import { 
  FileText, 
  Loader2, 
  CheckCircle, 
  AlertCircle, 
  RotateCcw,
  FolderCheck
} from 'lucide-react';

type WorkflowState = 
  | 'insert'
  | 'processing-front'
  | 'feedback-good'
  | 'feedback-reposition'
  | 'flip-page'
  | 'processing-back'
  | 'feedback-back-good'
  | 'complete';

export function ScanWorkflow() {
  const [state, setState] = useState<WorkflowState>('insert');
  const [pageNumber, setPageNumber] = useState(1);
  const [showTutorial, setShowTutorial] = useState(false);

  // Listen for Arduino button press (mapped to keyboard key 't')
  useEffect(() => {
    const handleKeyPress = (event: KeyboardEvent) => {
      if (event.key === 't' || event.key === 'T') {
        setShowTutorial(prev => !prev);
      }
    };

    window.addEventListener('keydown', handleKeyPress);
    return () => window.removeEventListener('keydown', handleKeyPress);
  }, []);

  // Auto-advance all states based on backend detection simulation
  useEffect(() => {
    let timer: NodeJS.Timeout;

    if (state === 'insert') {
      // Simulate backend detecting page insertion
      timer = setTimeout(() => {
        setState('processing-front');
      }, 5000);
    } else if (state === 'processing-front') {
      // Simulate backend processing and returning result
      timer = setTimeout(() => {
        const isGood = Math.random() > 0.3;
        setState(isGood ? 'feedback-good' : 'feedback-reposition');
      }, 2500);
    } else if (state === 'feedback-reposition') {
      // Wait for user to reposition, then auto-detect and start scanning
      timer = setTimeout(() => {
        setState('processing-front');
      }, 3000);
    } else if (state === 'feedback-good') {
      // Detect when page is flipped
      timer = setTimeout(() => {
        setState('flip-page');
      }, 2000);
    } else if (state === 'flip-page') {
      // Detect flip completed, start scanning back
      timer = setTimeout(() => {
        setState('processing-back');
      }, 2000);
    } else if (state === 'processing-back') {
      // Process back side
      timer = setTimeout(() => {
        setState('feedback-back-good');
      }, 2500);
    } else if (state === 'feedback-back-good') {
      // Show completion message
      timer = setTimeout(() => {
        setState('complete');
      }, 2000);
    } else if (state === 'complete') {
      // Auto-restart for next page
      timer = setTimeout(() => {
        setPageNumber(pageNumber + 1);
        setState('insert');
      }, 4000);
    }

    return () => clearTimeout(timer);
  }, [state, pageNumber]);

  return (
    <div className="min-h-screen flex items-center justify-center p-8">
      <div className="w-full max-w-3xl">
        {/* Page Counter */}
        <div className="text-center mb-6">
          <p className="text-4xl font-semibold text-gray-700">Page {pageNumber}</p>
        </div>

        {/* Main Status Card */}
        <div className={`bg-white rounded-3xl shadow-2xl p-12 min-h-[500px] flex flex-col items-center justify-center ${
          state === 'feedback-good' || state === 'feedback-back-good' 
            ? 'border-8 border-green-600' 
            : state === 'feedback-reposition' 
            ? 'border-8 border-orange-500' 
            : 'border-8 border-transparent'
        }`}>
          {state === 'insert' && (
            <div className="text-center space-y-8">
              <FileText className="w-32 h-32 mx-auto text-blue-600" strokeWidth={1.5} />
              <h1 className="text-5xl font-semibold text-gray-800">Insert Page</h1>
              <p className="text-3xl text-gray-600">Place the document on the scanner</p>
              <div className="text-center">
                <p className="text-2xl text-gray-500">Press the button for a tutorial video</p>
              </div>
            </div>
          )}

          {(state === 'processing-front' || state === 'processing-back') && (
            <div className="text-center space-y-8">
              <Loader2 className="w-32 h-32 mx-auto text-blue-600 animate-spin" strokeWidth={1.5} />
              <h1 className="text-5xl font-semibold text-gray-800">Scanning...</h1>
              <p className="text-3xl text-gray-600">
                {state === 'processing-front' ? 'Processing first side' : 'Processing second side'}
              </p>
            </div>
          )}

          {state === 'feedback-good' && (
            <div className="text-center space-y-8">
              <CheckCircle className="w-32 h-32 mx-auto text-green-600" strokeWidth={1.5} />
              <h1 className="text-5xl font-semibold text-green-700">Scan Successful!</h1>
              <p className="text-3xl text-gray-600">First side captured clearly</p>
              <p className="text-2xl text-gray-500 mt-4">Now flip the page over</p>
            </div>
          )}

          {state === 'feedback-reposition' && (
            <div className="text-center space-y-8">
              <AlertCircle className="w-32 h-32 mx-auto text-orange-500" strokeWidth={1.5} />
              <h1 className="text-5xl font-semibold text-orange-600">Needs Repositioning</h1>
              <p className="text-3xl text-gray-600">Please adjust the document</p>
              <p className="text-2xl text-gray-500 mt-4">Scanning will resume automatically</p>
            </div>
          )}

          {state === 'flip-page' && (
            <div className="text-center space-y-8">
              <div className="relative w-32 h-32 mx-auto">
                <FileText className="w-32 h-32 absolute text-blue-600 animate-pulse" strokeWidth={1.5} />
              </div>
              <h1 className="text-5xl font-semibold text-gray-800">Flip the Page</h1>
              <p className="text-3xl text-gray-600">Turn document over to scan the other side</p>
            </div>
          )}

          {state === 'feedback-back-good' && (
            <div className="text-center space-y-8">
              <CheckCircle className="w-32 h-32 mx-auto text-green-600" strokeWidth={1.5} />
              <h1 className="text-5xl font-semibold text-green-700">Second Side Complete!</h1>
              <p className="text-3xl text-gray-600">Both sides scanned successfully</p>
            </div>
          )}

          {state === 'complete' && (
            <div className="text-center space-y-8">
              <FolderCheck className="w-32 h-32 mx-auto text-blue-600" strokeWidth={1.5} />
              <h1 className="text-5xl font-semibold text-blue-700">Put Away File!</h1>
              <p className="text-3xl text-gray-600">Your document has been scanned</p>
              <p className="text-2xl text-gray-500 mt-4">You can now safely store your document</p>
            </div>
          )}
        </div>

        {/* Progress Indicator */}
        <div className="mt-8 text-center">
          <div className="inline-flex gap-3">
            <div className={`w-6 h-6 rounded-full ${
              ['insert', 'processing-front', 'feedback-good', 'feedback-reposition'].includes(state) 
                ? 'bg-blue-600' 
                : 'bg-gray-300'
            }`} />
            <div className={`w-6 h-6 rounded-full ${
              ['flip-page', 'processing-back', 'feedback-back-good'].includes(state) 
                ? 'bg-blue-600' 
                : 'bg-gray-300'
            }`} />
            <div className={`w-6 h-6 rounded-full ${
              state === 'complete' ? 'bg-blue-600' : 'bg-gray-300'
            }`} />
          </div>
        </div>
      </div>

      {/* Tutorial Modal */}
      {showTutorial && (
        <div className="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center p-8 z-50">
          <div className="bg-white rounded-3xl p-8 max-w-4xl w-full">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-4xl font-semibold text-gray-800">Tutorial Video</h2>
              <button
                onClick={() => setShowTutorial(false)}
                className="text-gray-600 hover:text-gray-800 text-5xl leading-none"
              >
                ×
              </button>
            </div>
            <div className="aspect-video bg-gray-900 rounded-xl flex items-center justify-center mb-6">
              {/* Replace this with your actual video */}
              <video
                controls
                className="w-full h-full rounded-xl"
                poster="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='800' height='450'%3E%3Crect fill='%23334155' width='800' height='450'/%3E%3Ctext fill='%23ffffff' font-size='24' font-family='sans-serif' x='50%25' y='50%25' text-anchor='middle' dominant-baseline='middle'%3ETutorial Video Placeholder%3C/text%3E%3C/svg%3E"
              >
                {/* Add your video source here */}
                <source src="YOUR_VIDEO_URL.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
            </div>
            <div className="text-center">
              <p className="text-2xl text-gray-500">Press the button to return to scanning</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}