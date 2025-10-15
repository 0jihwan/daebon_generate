import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QTextEdit, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
from docx import Document
from fpdf import FPDF
import os
import whisperx

from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))


class TranscriptionThread(QThread):
    """백그라운드에서 전사 처리"""
    finished = pyqtSignal(str)
    error = pyqtSignal(str)
    
    def __init__(self, audio_path, language):
        super().__init__()
        self.audio_path = audio_path
        self.language = language
    
    def run(self):
        temp_wav_path = None
        try:
            import tempfile
            import soundfile as sf
            
            device = "cpu"  # GPU 있으면 "cuda"
            compute_type = "int8"  # CPU 최적화
            
            # MP4/MP3를 WAV로 변환 (pyannote용)
            audio = whisperx.load_audio(self.audio_path)
            
            # 임시 WAV 파일 생성
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_wav_path = temp_wav.name
            temp_wav.close()
            
            # WAV로 저장 (16kHz, mono)
            sf.write(temp_wav_path, audio, 16000)
            
            # 1. Whisper 모델 로드 및 전사
            model = whisperx.load_model("large-v3", device, compute_type=compute_type, language=self.language)
            result = model.transcribe(audio, batch_size=16)
            
            # 2. 정렬 (단어 수준 타임스탬프)
            align_model, metadata = whisperx.load_align_model(
                language_code=self.language, 
                device=device
            )
            aligned_segments = whisperx.align(
                result["segments"], 
                align_model, 
                metadata, 
                audio, 
                device, 
                return_char_alignments=False
            )

            # result를 딕셔너리 형태로 유지
            result["segments"] = aligned_segments["segments"]
            result["word_segments"] = aligned_segments.get("word_segments", [])
            
            # 3. 화자 분리 (WAV 파일 사용)
            from pyannote.audio import Pipeline
            diarize_model = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=os.getenv("HF_TOKEN")
            )
            
            # device 설정
            if device == "cuda":
                import torch
                diarize_model.to(torch.device("cuda"))
            
            # WAV 파일로 화자 분리 실행
            diarization = diarize_model(temp_wav_path)
            
            # pyannote Annotation 객체를 pandas DataFrame 형태로 변환
            import pandas as pd
            diarize_df = pd.DataFrame([
                {'start': turn.start, 'end': turn.end, 'speaker': speaker}
                for turn, _, speaker in diarization.itertracks(yield_label=True)
            ])
            
            # whisperx에 화자 정보 할당
            result = whisperx.assign_word_speakers(diarize_df, result)
            
            # 임시 파일 삭제
            if temp_wav_path and os.path.exists(temp_wav_path):
                os.unlink(temp_wav_path)
            
            # 4. 대본 형식으로 변환
            transcript_lines = []
            current_speaker = None
            current_text = []
            
            for segment in result["segments"]:
                speaker = segment.get("speaker", "Unknown")
                text = segment["text"].strip()
                
                # 같은 화자면 텍스트 누적
                if speaker == current_speaker:
                    current_text.append(text)
                else:
                    # 화자 바뀌면 이전 내용 저장
                    if current_speaker is not None:
                        transcript_lines.append(f"{current_speaker}: {' '.join(current_text)}")
                    current_speaker = speaker
                    current_text = [text]
            
            # 마지막 화자 저장
            if current_speaker is not None:
                transcript_lines.append(f"{current_speaker}: {' '.join(current_text)}")
            
            final_text = "\n\n".join(transcript_lines)
            self.finished.emit(final_text)
            
        except Exception as e:
            # 임시 파일 정리
            if temp_wav_path and os.path.exists(temp_wav_path):
                try:
                    os.unlink(temp_wav_path)
                except:
                    pass

            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n상세 정보:\n{traceback.format_exc()}"
            self.error.emit(error_msg)





class TranscriptApp(QWidget):
    def __init__(self):
        super().__init__()
        self.audio_file_path = None
        self.transcription_thread = None
        self.initUI()
    
    def initUI(self):
        self.setWindowTitle('다국어 대본 작성 프로그램')
        self.setGeometry(100, 100, 600, 450)

        self.label = QLabel('음성 파일 선택:', self)
        self.label.move(20, 20)

        self.btn = QPushButton('파일 열기', self)
        self.btn.move(120, 20)
        self.btn.clicked.connect(self.openFile)

        self.lang_label = QLabel('언어 선택:', self)
        self.lang_label.move(20, 60)

        self.lang_combo = QComboBox(self)
        self.lang_combo.addItems(['한국어', '영어', '일본어'])
        self.lang_combo.move(120, 60)

        self.status_label = QLabel('대기 중...', self)
        self.status_label.setGeometry(20, 100, 560, 20)

        self.transcript = QTextEdit(self)
        self.transcript.setGeometry(20, 130, 560, 200)

        self.save_pdf_btn = QPushButton('PDF로 저장', self)
        self.save_pdf_btn.move(20, 350)
        self.save_pdf_btn.clicked.connect(self.savePDF)

        self.save_docx_btn = QPushButton('DOCX로 저장', self)
        self.save_docx_btn.move(120, 350)
        self.save_docx_btn.clicked.connect(self.saveDOCX)

    def openFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, '파일 선택', '', 'Audio Files (*.wav *.mp3 *.mp4)')
        if not fname:
            return
        
        self.audio_file_path = fname
        
        # 언어 코드 매핑
        lang_map = {'한국어': 'ko', '영어': 'en', '일본어': 'ja'}
        language = lang_map[self.lang_combo.currentText()]
        
        # 상태 표시
        self.status_label.setText('처리 중... (시간이 다소 걸릴 수 있습니다)')
        self.transcript.setText('')
        self.btn.setEnabled(False)
        
        # 백그라운드 스레드에서 처리
        self.transcription_thread = TranscriptionThread(fname, language)
        self.transcription_thread.finished.connect(self.onTranscriptionComplete)
        self.transcription_thread.error.connect(self.onTranscriptionError)
        self.transcription_thread.start()
    
    def onTranscriptionComplete(self, text):
        self.transcript.setText(text)
        self.status_label.setText('완료!')
        self.btn.setEnabled(True)
    
    def onTranscriptionError(self, error_msg):
        self.transcript.setText(f"오류 발생:\n{error_msg}")
        self.status_label.setText('오류 발생')
        self.btn.setEnabled(True)

    def savePDF(self):
        text = self.transcript.toPlainText()
        fname, _ = QFileDialog.getSaveFileName(self, 'PDF로 저장', '', 'PDF Files (*.pdf)')
        if fname:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, text)
            pdf.output(fname)

    def saveDOCX(self):
        text = self.transcript.toPlainText()
        fname, _ = QFileDialog.getSaveFileName(self, 'DOCX로 저장', '', 'Word Files (*.docx)')
        if fname:
            doc = Document()
            doc.add_paragraph(text)
            doc.save(fname)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TranscriptApp()
    ex.show()
    sys.exit(app.exec_())
