package main

import (
	"encoding/json"
	"fmt"
	"os"
)

func EncodeAuthorsToJSONLines(authors Authors) {
	for _, author := range authors.List {
		authorJSON := AuthorToJSON(author)
		jsonData, err := json.Marshal(authorJSON)
		if err != nil {
			panic(err)
		}
		if _, err = os.Stdout.Write(jsonData); err != nil {
			panic(err)
		}
		fmt.Println()
	}
}
