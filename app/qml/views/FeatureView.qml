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
                Layout.preferredHeight: 140
                radius: 12
                color: "#111C2D"
                border.color: "#1A2A40"
                Column {
                    anchors.fill: parent
                    anchors.margins: 12
                    spacing: 6
                    Label { text: "Feature Lab"; color: "#E2EEFF"; font.pixelSize: 20; font.bold: true }
                    Label { text: "Generated features: " + appState.generatedFeatureCount; color: "#9BB6D8" }
                    Label { text: "Rows after generation: " + appState.featureRowCount; color: "#9BB6D8" }
                    Button { text: "Generate Features"; onClicked: appState.generateFeatures() }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 140
                radius: 12
                color: "#0F1725"
                border.color: "#1B2A41"
                ScrollView {
                    anchors.fill: parent
                    anchors.margins: 10
                    TextArea {
                        readOnly: true
                        wrapMode: Text.Wrap
                        color: "#BFD3EC"
                        text: {
                            var cols = appState.featureColumns || []
                            if (cols.length === 0) return "No generated feature columns yet"
                            return cols.join(", ")
                        }
                    }
                }
            }

            Rectangle {
                Layout.fillWidth: true
                Layout.preferredHeight: 320
                radius: 12
                color: "#0F1725"
                border.color: "#1B2A41"
                Column {
                    anchors.fill: parent
                    anchors.margins: 10
                    spacing: 6
                    Label { text: "Preview (first 20 rows of feature dataset)"; color: "#DCEAFF"; font.bold: true }
                    ScrollView {
                        anchors.left: parent.left
                        anchors.right: parent.right
                        anchors.top: parent.top
                        anchors.bottom: parent.bottom
                        anchors.topMargin: 26
                        TextArea {
                            readOnly: true
                            wrapMode: Text.NoWrap
                            color: "#BFD3EC"
                            text: {
                                var cols = appState.featureColumns || []
                                if (cols.length === 0) return "No feature dataset generated"
                                var rows = appState.featurePreviewRows || []
                                var lines = [cols.join(" | ")]
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
        }
    }
}
