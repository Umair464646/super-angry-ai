import QtQuick
import QtQuick.Controls
import QtQuick.Layouts

Item {
    Layout.fillWidth: true
    Layout.fillHeight: true

    ScrollView {
        anchors.fill: parent
        contentWidth: availableWidth

        ColumnLayout {
            width: parent.width
            spacing: 10

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 190
                radius: 12
                color: "#111C2D"
                border.color: "#1A2A40"
                Column {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 6
                    Label { text: "Data Lab"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }
                    Label { text: "Path: " + (appState.profile.path || "No dataset loaded"); color: "#9BB6D8"; elide: Text.ElideMiddle; width: parent.width }
                    Label { text: "Rows: " + (appState.profile.rows || 0); color: "#9BB6D8" }
                    Label { text: "Columns: " + ((appState.profile.columns || []).join(", ")); color: "#9BB6D8"; width: parent.width; wrapMode: Text.Wrap }
                    Label { text: "Range: " + (appState.profile.start || "n/a") + " -> " + (appState.profile.end || "n/a"); color: "#9BB6D8"; width: parent.width; elide: Text.ElideRight }
                    Label { text: "Synthetic %: " + Number(appState.profile.synthetic_pct || 0).toFixed(2); color: "#8FD3FF" }
                    Button { text: "Clear Dataset"; onClicked: appState.clearDataset() }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 300
                radius: 12
                color: "#0F1725"
                border.color: "#1B2A41"
                ColumnLayout {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 6
                    Label { text: "Preview (first 20 rows)"; color: "#DCEAFF"; font.bold: true }
                    ScrollView {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        TextArea {
                            readOnly: true
                            wrapMode: Text.NoWrap
                            color: "#BFD3EC"
                            text: {
                                var cols = appState.previewColumns || []
                                if (cols.length === 0)
                                    return "No dataset loaded"
                                var lines = [cols.join(" | ")]
                                var rows = appState.previewRows || []
                                for (var i = 0; i < rows.length; ++i) {
                                    var r = rows[i]
                                    var vals = []
                                    for (var c = 0; c < cols.length; ++c)
                                        vals.push(r[cols[c]])
                                    lines.push(vals.join(" | "))
                                }
                                return lines.join("\n")
                            }
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.fillHeight: true
                radius: 12
                color: "#0F1725"
                border.color: "#1B2A41"
                ListView {
                    anchors.fill: parent
                    anchors.margins: 10
                    model: appState.logs
                    delegate: Label {
                        required property var modelData
                        text: "[" + modelData.ts + "] " + modelData.msg
                        color: "#9EC0E3"
                        wrapMode: Text.Wrap
                    }
                    ScrollBar.vertical: ScrollBar {}
                }
            }
        }
    }
}
