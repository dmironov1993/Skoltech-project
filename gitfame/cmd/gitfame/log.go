package main

import (
	"bufio"
	"bytes"
	"os/exec"
	"strings"
)

// cmd := exec.Command("myCommand", "arg1", "arg2")
// cmd.Dir = "/path/to/work/dir"
// cmd.Run()

func Log(repository, revision, fileName string) (hash string, author string) {
	args := []string{"log", "--format=%H %an", revision, "--", fileName}
	cmd := exec.Command("git", args...)
	cmd.Dir = repository

	output, err := cmd.Output()
	if err != nil {
		return "", ""
	}

	buf := bytes.Buffer{}
	buf.Write(output)
	scanner := bufio.NewScanner(&buf)

	if !scanner.Scan() {
		return "", ""
	}

	entries := strings.SplitN(scanner.Text(), " ", 2)
	if len(entries) < 2 {
		return "", ""
	}

	hash = entries[0]
	author = entries[1]
	return
}
