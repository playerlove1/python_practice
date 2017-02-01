#檢查文字檔內容是否有不雅字串
import urllib.request
import urllib.parse
#讀檔函式
def read_text():
	#使用相對路徑開啟檔案  (open 為python build in的函式)
	qupotes=open("05-movie_quotes.txt")
	#使用檔案.read() 取得檔案內容
	contents_of_file=qupotes.read()
	#印出結果
	print(contents_of_file)
	#關閉檔案
	qupotes.close()
	#將檔案傳入檢查的函式
	check_profanity(contents_of_file)
	
	
#檢查文字中是否有不雅用詞
def check_profanity(text_to_check):
	# 透過google的 what do you love 會透過q= 查詢字串 會回傳 結果
	#wdyl的url  原URL:http://www.wdyl.com/profanity  已經被deprecated
	#因此採用udacity課程提供的URL
	wdyl_url="http://www.wdylike.appspot.com/"
	#透過傳入q= text_to_check 來檢查字串是否有不雅用詞
	#由於傳入的文件字串當中的換行'/n' 無法被正確解析  因此使用urllib.parse.quote 將後方的參數已URL接受的編碼做轉換
	query_url=wdyl_url+'?q='+urllib.parse.quote(text_to_check)
	#傳入該URL	
	#print(query_url)
	connection = urllib.request.urlopen(query_url)

	#取得該URL回傳的結果
	output= connection.read().decode('utf-8')
	#印出結果
	print(output)
	#關閉連線
	connection.close()
	if "true" in output:
		print("Profanity Alert!!")
	elif "false" in output:
		print("This is document has no curse words!")
	else:
		print("Could not scan the document properly.")
read_text()
