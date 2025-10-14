import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QFileDialog, QComboBox, QTextEdit
from transformers import pipeline
from docx import Document
from fpdf import FPDF
import os

from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
login(token=os.getenv("HF_TOKEN"))

class TranscriptApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.audio_file_path = None
        self.pipelines = {
            '한국어': pipeline("automatic-speech-recognition", model="bigdefence/Bigvox-HyperCLOVAX-Audio"),
            '영어': pipeline("automatic-speech-recognition", model="openai/whisper-large-v3"),
            '일본어': pipeline("automatic-speech-recognition", model="kotoba-tech/kotoba-whisper-v1.0")
        }
        self.selected_lang = '한국어'

    def initUI(self):
        self.setWindowTitle('다국어 대본 작성 프로그램')
        self.setGeometry(100, 100, 600, 400)

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
        self.lang_combo.currentTextChanged.connect(self.set_lang)

        self.transcript = QTextEdit(self)
        self.transcript.setGeometry(20, 100, 560, 200)

        self.save_pdf_btn = QPushButton('PDF로 저장', self)
        self.save_pdf_btn.move(20, 320)
        self.save_pdf_btn.clicked.connect(self.savePDF)

        self.save_docx_btn = QPushButton('DOCX로 저장', self)
        self.save_docx_btn.move(120, 320)
        self.save_docx_btn.clicked.connect(self.saveDOCX)

    def set_lang(self, lang):
        self.selected_lang = lang

    def openFile(self):
        fname, _ = QFileDialog.getOpenFileName(self, '파일 선택', '', 'Audio Files (*.wav *.mp3 *.mp4)')
        if not fname:
            return
        self.audio_file_path = fname
        pipe = self.pipelines[self.selected_lang]
        result = pipe(fname)
        text = result['text'] if 'text' in result else ''
        self.transcript.setText(text)

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
