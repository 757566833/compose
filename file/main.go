package main

import (
	"file/db"
	"file/log"
	"file/route"
	"github.com/joho/godotenv"
	"os"
)

func main() {
	log.InitLogger()
	err := godotenv.Load(".env")
	if err != nil {
		os.Exit(1)
	}
	db.InitDB()
	ExplorerServerPort := os.Getenv("EXPLORER_SERVER_PORT")
	//ExplorerServerPort := "8090"
	router := route.InitRouter()
	err = router.Run("0.0.0.0:" + ExplorerServerPort)
	if err != nil {
		log.Logger.Error(err.Error())
	}
}
