#define AppName "Multi-Agent Paper Reader"
#ifndef AppVersion
#define AppVersion "1.2.0"
#endif
#define AppPublisher "Multi-Agent Paper Reader"
#define AppExeName "PaperReader.exe"

[Setup]
AppId={{EA8D2BB2-4F45-4AA1-B7E8-15669B88E911}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL=https://github.com/Lovenndme/multi-agent-paper-reader
AppSupportURL=https://github.com/Lovenndme/multi-agent-paper-reader/issues
DefaultDirName={localappdata}\Programs\Multi-Agent Paper Reader
DefaultGroupName=Multi-Agent Paper Reader
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\release
OutputBaseFilename=Multi-Agent-Paper-Reader-{#AppVersion}-Windows-x64-Setup
SetupIconFile=assets\paper-reader.ico
UninstallDisplayIcon={app}\{#AppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=110
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=force
RestartApplications=no
#ifdef SigningEnabled
SignedUninstaller=yes
SignTool=preview
#endif
VersionInfoVersion={#AppVersion}.0
VersionInfoCompany={#AppPublisher}
VersionInfoDescription={#AppName} Installer
VersionInfoProductName={#AppName}
VersionInfoProductVersion={#AppVersion}
SetupLogging=yes

[Languages]
Name: "chinesesimp"; MessagesFile: "languages\ChineseSimplified.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："; Flags: checkedonce

[Files]
Source: "dist\PaperReader\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\Multi-Agent Paper Reader"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"
Name: "{group}\卸载 Multi-Agent Paper Reader"; Filename: "{uninstallexe}"
Name: "{autodesktop}\Multi-Agent Paper Reader"; Filename: "{app}\{#AppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "启动 Multi-Agent Paper Reader"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\{#AppExeName}"; Parameters: "--shutdown-for-uninstall"; Flags: runhidden waituntilterminated; RunOnceId: "StopPaperReader"

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
end;
