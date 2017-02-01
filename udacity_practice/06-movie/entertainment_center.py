#引用先前寫好的media.py檔案
import media
import fresh_tomatoes

#建立new instance 由media.py中 呼叫Movie類別
#self 會default 被加入所以在new instance時可以忽略
toy_story = media.Movie("Toy Story", "A story of a boy and his toys that come to life", "https://upload.wikimedia.org/wikipedia/en/1/13/Toy_Story.jpg", "https:/www.youtube.com/watch?v=vwyZH85NQC4")
	#初始化過程步驟
	#1.__init__ 初始化函數被呼叫
	#2.self=toy_sotry
	#3.movie_title="Toy Story"
	#4.movie_storyline="A story of a boy and his toys that come to life"
	#5.poster_image="http://upload.wikimedia.org/wikipedia/en/1/13/ToyStory.jpg"
	#6.trailer_youtube="https:/www.youtube.com/watch?v=vwyZH85NQC4"
avatar = media.Movie("Avatar", "A marline on an alien planet", "http://upload.wikimedia.org/wikipedia/id/b/b0/Avatar-Teaser-Poster.jpg", "http://www.youtube.com/watch?v=-9ceBgWV8io")

school_of_rock = media.Movie("School of Rock", "Storyline", "https://upload.wikimedia.org/wikipedia/en/1/11/School_of_Rock_Poster.jpg", "https://www.youtube.com/watch?v=3PsUJFEBC74")

ratatouille = media.Movie("Ratatouille", "Storyline", "https://upload.wikimedia.org/wikipedia/en/5/50/RatatouillePoster.jpg", "https://www.youtube.com/watch?v=c3sBBRxDAqk")

midnight_in_paris = media.Movie("Midnight in Paris", "Storyline", "https://upload.wikimedia.org/wikipedia/en/9/9f/Midnight_in_Paris_Poster.jpg","https://www.youtube.com/watch?v=atLg2wQQxvU")

hunger_games = media.Movie("Hunger Games", "Storyline","https://upload.wikimedia.org/wikipedia/en/4/42/HungerGamesPoster.jpg","https://www.youtube.com/watch?v=PbA63a7H0bo" )

#將所有電影放入list
movies= [toy_story, avatar, school_of_rock, ratatouille, midnight_in_paris, hunger_games]
#顯示電影頁面
fresh_tomatoes.open_movies_page(movies)

#print(toy_story.storyline)
#avatar.show_trailer()