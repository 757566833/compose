package services

import (
	"bytes"
	"context"
	"encoding/json"
	"file/db"
	"file/utils"
	"fmt"
	"github.com/gin-gonic/gin"
	"github.com/minio/minio-go/v7"
	"net/http"
	"strings"
)

const Region = "local"

const contentJSONType = "application/json"

func checkBuket(buket string) error {
	exists, err := db.MinIoClient.BucketExists(context.Background(), buket)
	if err == nil && exists {
		// 已存在
	} else {
		op := new(minio.MakeBucketOptions)
		op.Region = Region

		err = db.MinIoClient.MakeBucket(context.Background(), buket, *op)
		if err != nil {
			return err
		}
	}
	return nil
}

func Upload(c *gin.Context) {
	buket := c.Param("buket")
	err := checkBuket(buket)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	f, err := c.FormFile("file")
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	fileIo, err := f.Open()
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	_type, err := utils.GetFileContentType(fileIo)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}

	defer db.Close(fileIo)

	name := utils.RandStr(32)
	split := strings.Split(f.Filename, ".")
	fmt.Println(split)

	if len(split) > 1 {
		name = name + "." + split[len(split)-1]
	}
	_, err = db.MinIoClient.PutObject(context.Background(), buket, name, fileIo, f.Size, minio.PutObjectOptions{ContentType: _type, UserTags: map[string]string{"filename": f.Filename}})
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	c.IndentedJSON(http.StatusOK, "/preview/"+buket+"/"+name)
}

func Preview(c *gin.Context) {
	buket := c.Param("buket")
	file := c.Param("file")
	if file == "" {
		c.IndentedJSON(http.StatusBadRequest, "")
	}

	object, err := db.MinIoClient.GetObject(context.Background(), buket, file, minio.GetObjectOptions{})
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	objectInfo, err := object.Stat()
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	defer db.Close(object)
	extraHeaders := map[string]string{
		"Content-Disposition": "inline",
	}
	for key, value := range objectInfo.Metadata {
		extraHeaders[key] = strings.Join(value, ";")
	}
	c.DataFromReader(http.StatusOK, objectInfo.Size, objectInfo.Metadata.Get("Content-Type"), object, extraHeaders)

}
func Download(c *gin.Context) {
	buket := c.Param("buket")
	file := c.Param("file")
	if file == "" {
		c.IndentedJSON(http.StatusBadRequest, "")
	}
	object, err := db.MinIoClient.GetObject(context.Background(), buket, file, minio.GetObjectOptions{})
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	objectInfo, err := object.Stat()
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}

	defer db.Close(object)

	extraHeaders := map[string]string{
		"Content-Disposition": "attachment; filename=" + file,
	}
	for key, value := range objectInfo.Metadata {
		extraHeaders[key] = strings.Join(value, ";")
	}
	c.DataFromReader(http.StatusOK, objectInfo.Size, objectInfo.Metadata.Get("Content-Type"), object, extraHeaders)
}

func All(c *gin.Context) {
	buket := c.Param("buket")
	object, err := db.MinIoCore.ListObjects(buket, "", "", "", 1000)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}

	c.JSON(http.StatusOK, object.Contents)
}

type CreateJsonStruct struct {
	Name string `json:"name"`
	Data any    `json:"data"`
}

func CreateJson(c *gin.Context) {
	buket := c.Param("buket")
	err := checkBuket(buket)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	var requestBody CreateJsonStruct
	if err := c.BindJSON(&requestBody); err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	name := requestBody.Name
	_object, err := db.MinIoClient.GetObject(context.Background(), buket, name, minio.GetObjectOptions{})
	defer db.Close(_object)
	if err == nil && _object != nil {
		c.IndentedJSON(http.StatusBadRequest, name+" existed")
		return
	}
	data := requestBody.Data
	str, err := json.Marshal(data)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	_data := bytes.NewReader(str)

	object, err := db.MinIoClient.PutObject(context.Background(), buket, name, _data, int64(len(str)), minio.PutObjectOptions{ContentType: contentJSONType})

	if err != nil {
		fmt.Println(err.Error())
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	c.IndentedJSON(http.StatusOK, object)
}
func ForceJson(c *gin.Context) {
	buket := c.Param("buket")
	err := checkBuket(buket)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	var requestBody CreateJsonStruct
	if err := c.BindJSON(&requestBody); err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	name := requestBody.Name
	data := requestBody.Data
	str, err := json.Marshal(data)
	if err != nil {
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	_data := bytes.NewReader(str)

	object, err := db.MinIoClient.PutObject(context.Background(), buket, name, _data, int64(len(str)), minio.PutObjectOptions{ContentType: contentJSONType})
	if err != nil {
		fmt.Println(err.Error())
		c.IndentedJSON(http.StatusBadRequest, err.Error())
	}
	c.IndentedJSON(http.StatusOK, object)
}
