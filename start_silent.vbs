Set WshShell = CreateObject("WScript.Shell")
WshShell.Run chr(34) & Replace(WScript.ScriptFullName, "start_silent.vbs", "Run_AI_Enhancer.bat") & Chr(34), 0
Set WshShell = Nothing
