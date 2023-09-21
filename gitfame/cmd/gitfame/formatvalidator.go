package main

func ValidateFormat(format string) {
	allowedFormats := []string{
		"tabular",
		"csv",
		"json",
		"json-lines",
	}
	for _, f := range allowedFormats {
		if format == f {
			return
		}
	}
	panic("format flag value is incorrect")
}
