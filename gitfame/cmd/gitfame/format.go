package main

// before running this function 'format' has already been validated by 'formatvalidator.go'

type FormatterStrategy func(author Authors)

var Strategies = map[string]FormatterStrategy{
	"tabular":    EncodeAuthorsToTabular,
	"csv":        EncodeAuthorsToCSV,
	"json":       EncodeAuthorsToJSON,
	"json-lines": EncodeAuthorsToJSONLines,
}

func Format(authors Authors, format string) {
	if strategy, found := Strategies[format]; found {
		strategy(authors)
	}
}
