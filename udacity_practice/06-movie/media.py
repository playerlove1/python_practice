#google python style guide (命名風格):https://google.github.io/styleguide/pyguide.html
import webbrowser
class Movie():
	#類別的說明文件  可以透過 類別.__doc__取得內容  由 前後各3個""" 夾住 (可多行)
	"""This calss provides a way to store movie related information
	    這是一個用來提供儲存電影相關資訊的類別
	"""
	#類別變數
	#配合google python stytle 此變數為常數則將此變數的名稱都變為大寫
	VAILD_RATINGS = ["G", "PG", "PG1-3", "R"]
	#建構式  傳入的參數維本身self
	def __init__(self, movie_title, movie_storyline, poster_image, trailer_youtube):
		#instance Variables
		self.title = movie_title
		self.storyline = movie_storyline
		self.poster_image_url = poster_image
		self.trailer_youtube_url = trailer_youtube
	#Instance Methods
	def show_trailer(self):
		webbrowser.open(self.trailer_youtube_url)
		