package main

import (
	"github.com/spf13/pflag"
	"os"
)

type FlagParameters struct {
	value   any
	pointer string
	usage   string
}

var Flags = []string{
	"repository",
	"revision",
	"order-by",
	"use-committer",
	"format",
	"extensions",
	"languages",
	"exclude",
	"restrict-to",
}

var FlagOptions = map[string]FlagParameters{
	"repository": {
		pointer: ".",
		usage:   "path to Git repository; a current directory is by default",
	},
	"revision": {
		pointer: "HEAD",
		usage:   "pointer to commit; HEAD is by default",
	},
	"order-by": {
		pointer: "lines",
		usage:   "a sorting key; lines by default, but can be 'commits' or 'files'",
	},
	"use-committer": {
		value: false,
		usage: "a boolean flag changing an author to committer",
	},
	"format": {
		pointer: "tabular",
		usage:   "output format; tabular is by default, but can be 'csv', 'json', 'json-lines'",
	},
	"extensions": {
		value: []string{},
		usage: "list of extensions shrinking a list of files in calculations; " +
			"set of restrictions divided by commas, for example '.go,.md'",
	},
	"languages": {
		value: []string{},
		usage: "list of languages shrinking a list of files in calculations; " +
			"set of restrictions divided by commas, for example 'go,markdown'",
	},
	"exclude": {
		value: []string{},
		usage: "set of Glob patterns excluding files from calculations, example: \\'foo/*,bar/*'\\",
	},
	"restrict-to": {
		value: []string{},
		usage: "set of Glob patterns excluding all files not satisfying any pattern",
	},
}

// ParseArguments https://github.com/spf13/pflag/issues/195
func ParseArguments() {
	cmd := pflag.NewFlagSet("command", pflag.ExitOnError)

	for _, flag := range Flags {
		flagOptions := FlagOptions[flag]
		switch flagOptions.value.(type) {
		case []string:
			cmd.StringSlice(flag, []string{}, flagOptions.usage)
		case bool:
			cmd.Bool(flag, false, flagOptions.usage)
		default:
			cmd.String(flag, flagOptions.pointer, flagOptions.usage)
		}
	}

	err := cmd.Parse(os.Args[1:])
	if err != nil {
		panic("MAIN: os.Args[1:] generates an error")
	}

	// parsed values in the task description order
	repository, _ = cmd.GetString("repository")
	revision, _ = cmd.GetString("revision")
	orderby, _ = cmd.GetString("order-by")
	committer, _ = cmd.GetBool("use-committer")
	format, _ = cmd.GetString("format")
	extensions, _ = cmd.GetStringSlice("extensions")
	languages, _ = cmd.GetStringSlice("languages")
	exclude, _ = cmd.GetStringSlice("exclude")
	restrict, _ = cmd.GetStringSlice("restrict-to")
}
