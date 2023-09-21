package main

import (
	"encoding/json"
	"os"
)

func EncodeAuthorsToJSON(authors Authors) {
	authorsJSON := make([]JSONFields, len(authors.List))
	for index, author := range authors.List {
		authorsJSON[index] = AuthorToJSON(author)
	}
	jsonData, err := json.Marshal(&authorsJSON)
	if err != nil {
		panic(err)
	}
	if _, err = os.Stdout.Write(jsonData); err != nil {
		panic(err)
	}
}
