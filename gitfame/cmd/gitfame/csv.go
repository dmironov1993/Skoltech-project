package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
)

func writeToCSV(writer *csv.Writer, record []string) error {
	err := writer.Write(record)
	if err != nil {
		return fmt.Errorf("CSV: w.Write generates an error: %w", err)
	}
	return nil
}

func writeHeaderToCSV(writer *csv.Writer) error {
	return writeToCSV(writer, []string{"Name", "Lines", "Commits", "Files"})
}

func writeAuthorToCSV(writer *csv.Writer, author *Author) error {
	return writeToCSV(writer, []string{
		author.Name,
		strconv.Itoa(author.Stats.Lines),
		strconv.Itoa(author.Stats.Commits),
		strconv.Itoa(author.Stats.Files),
	})
}

func EncodeAuthorsToCSV(authors Authors) {
	w := csv.NewWriter(os.Stdout)
	defer func() {
		w.Flush()
		if err := w.Error(); err != nil {
			fmt.Printf("Error flushing CSV writer: %s\n", err)
		}
	}()

	if err := writeHeaderToCSV(w); err != nil {
		panic(err)
	}

	for _, author := range authors.List {
		if err := writeAuthorToCSV(w, &author); err != nil {
			panic(err)
		}
	}

}
