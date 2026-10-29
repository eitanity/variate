package st

func stmts() (err error) {
	defer func() {
		if r := recover(); r != nil { // want "recover outside the root package"
			err = nil
		}
	}()
	go func() {}() // want "go statement outside the root package"
	goto end       // want "goto"
end:
	return nil
}
